import json
import logging
import re
import unicodedata
from typing import Dict, List
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from src.company_detail.schema import (
    AddressOutput,
    BusinessSummaryOutput,
    CompanyDetailOutput,
)
from src.infra.langfuse import WithSpanContext, with_langfuse_span
from src.infra.llm import generate_structured_output
from src.infra.llm.registry import ModelName

from ..extract import PageExtractionResult

logger = logging.getLogger(__name__)


class MergeAddressOutput(BaseModel):
    description: str = Field(
        ..., description="Address label such as 本社, 支社, 営業所"
    )
    address: str = Field(..., description="Raw address text as written in source page")
    sourceSlot: int = Field(
        ..., ge=1, description="1-based slot number of evidence page"
    )


class CitationSlotItem(BaseModel):
    citation: str = Field(
        ...,
        description="Citation number string used in detail text, e.g. '1' for [1]",
    )
    sourceSlot: int = Field(
        ..., ge=1, description="1-based slot number of evidence page"
    )


class MergeBusinessSummaryOutput(BaseModel):
    detail: str = Field(
        ...,
        description="Japanese business summary text with citations like [1], [2]",
    )
    citationSlots: List[CitationSlotItem] = Field(
        ...,
        description="Citation to source-slot mapping list",
    )


class MergeStructuredOutput(BaseModel):
    address: List[MergeAddressOutput]
    business_summary: MergeBusinessSummaryOutput


def merge_company_detail_extractions(
    company_name: str,
    company_url: str,
    extractions: List[PageExtractionResult],
    *,
    model: ModelName = "openai/gpt-5.4-mini",
    span_context: WithSpanContext | None = None,
) -> CompanyDetailOutput:
    """
    フロー3: 統合 (Merge)

    複数のページからの抽出結果を統合し、最終的なJSON形式を生成する。
    - 同一ドメイン情報を優先
    - 本社を先頭に配置
    - 参照番号の割り当て

    Args:
        company_name (str): 企業名
        company_url (str): 企業URL
        extractions (List[PageExtractionResult]): ページごとの抽出結果リスト

    Returns:
        CompanyDetailOutput: 統合された最終結果
    """

    with with_langfuse_span(
        span_name="validate_and_finalize_profile",
        span_context=span_context,
    ) as span:
        span.set_input(
            {
                "company_name": company_name,
                "company_url": company_url,
                "num_extractions": len(extractions),
            }
        )

        slot_to_url: Dict[int, str] = {}
        pages_for_prompt = []

        for index, extraction in enumerate(extractions, start=1):
            slot_to_url[index] = extraction.url

            parsed_url = urlparse(extraction.url)
            path_hint = parsed_url.path or "/"
            pages_for_prompt.append(
                {
                    "urlSlot": index,
                    "title": extraction.title,
                    "pathHint": path_hint,
                    "business": extraction.extracted.business,
                    "addresses": [
                        {
                            "description": address.description,
                            "address": address.address,
                        }
                        for address in extraction.extracted.addresses
                    ],
                }
            )

        merge_prompt = f"""
# Input
- company_name: {company_name}
- company_url: {company_url}

## page_extractions
{json.dumps(pages_for_prompt, ensure_ascii=False, indent=2)}
"""

        merged = generate_structured_output(
            model=model,
            system_prompt="""
役割:
- ページ単位の抽出結果を統合し、最終的な会社情報JSONを作成してください。

必須ルール:
- 出力スキーマに一致するJSONだけを返してください。
- Markdown、説明文、余分なキーは不要です。
- URL文字列を直接出力しないでください。根拠は必ず sourceSlot / citationSlots で参照してください。
- 提供された page_extractions の情報だけを根拠にしてください。
- page_extractionsにない住所・事業内容を推測しないでください。

住所の統合ルール:
- 住所は最大5件です。
- 本社、本店、本社オフィスを優先して先頭に置いてください。
- その後に主要な支社、営業所、拠点を入れてください。
- 各住所には根拠ページを示す sourceSlot を必ず入れてください。
- 重複・類似する住所はまとめてください。
- addressは入力された住所候補の文字列を短縮しないでください。建物名、ビル名、施設名、階数、部屋番号、郵便番号が含まれている場合は保持してください。
- 同じ所在地に短い住所と長い住所がある場合は、建物名や階数まで含む長い住所を優先してください。

事業概要の統合ルール:
- business_summary.detail は日本語で書いてください。
- 公式サイトに書かれた事業、サービス、プロダクト、ソリューションの事実を中心に要約してください。
- 本文中には `[1]`, `[2]` のような引用番号を含めてください。
- citationSlots は本文中の引用番号と sourceSlot の対応だけを入れてください。
- 有効な事業根拠がない場合は detail を空文字、citationSlots を空配列にしてください。

フォールバック:
- 有効な住所根拠がない場合、address は空配列にしてください。

出力例:
- address の例:
    {
        "address": [
            {"description": "本社", "address": "東京都千代田区...", "sourceSlot": 1},
            {"description": "支社", "address": "大阪府大阪市...", "sourceSlot": 2}
        ]
    }
- business_summary の例:
    {
        "business_summary": {
            "detail": "主力事業はデータ分析基盤の提供。[1] 金融向けソリューションも展開。[2]",
            "citationSlots": [
                {"citation": "1", "sourceSlot": 1},
                {"citation": "2", "sourceSlot": 2}
            ]
        }
    }
""",
            prompt=merge_prompt,
            output_schema=MergeStructuredOutput,
            generation_name="merge_company_profile",
            metadata={
                "num_pages_used": len(extractions),
                "num_address_candidates": sum(
                    len(item.extracted.addresses) for item in extractions
                ),
                "num_business_candidates": sum(
                    len(item.extracted.business) for item in extractions
                ),
            },
            parent_span=span.span,
        )

        final_addresses = _postprocess_addresses(merged.address, slot_to_url)
        final_business_summary = _build_business_summary(
            merged.business_summary.detail,
            merged.business_summary.citationSlots,
            slot_to_url,
        )

        return span.finish(
            CompanyDetailOutput(
                company_name=company_name,
                company_url=company_url,
                address=final_addresses,
                business_summary=final_business_summary,
                viewed_source_urls=list(slot_to_url.values()),
            )
        )


def _normalize_for_dedupe(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip()
    normalized = re.sub(r"[‐‑‒–—―ー−]", "-", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _is_hq(description: str) -> bool:
    return "本社" in description


def _build_business_summary(
    detail: str,
    citation_slots: List[CitationSlotItem],
    slot_to_url: Dict[int, str],
) -> BusinessSummaryOutput:
    cleaned_detail = detail.strip()
    if not cleaned_detail:
        return BusinessSummaryOutput(detail="", sourceUrls={})

    cited_keys = set(re.findall(r"\[(\d+)\]", cleaned_detail))
    if not cited_keys:
        return BusinessSummaryOutput(detail="", sourceUrls={})

    citation_map = {item.citation: item.sourceSlot for item in citation_slots}
    valid_source_urls = {
        key: slot_to_url[slot]
        for key, slot in citation_map.items()
        if key in cited_keys and slot in slot_to_url
    }
    if not valid_source_urls:
        return BusinessSummaryOutput(detail="", sourceUrls={})

    cleaned_detail = re.sub(
        r"\[(\d+)\]",
        lambda match: match.group(0) if match.group(1) in valid_source_urls else "",
        cleaned_detail,
    )
    cleaned_detail = re.sub(r"\s{2,}", " ", cleaned_detail).strip()
    ordered_source_urls = {
        key: valid_source_urls[key]
        for key in sorted(valid_source_urls.keys(), key=lambda k: int(k))
    }
    if not cleaned_detail:
        return BusinessSummaryOutput(detail="", sourceUrls={})

    return BusinessSummaryOutput(detail=cleaned_detail, sourceUrls=ordered_source_urls)


def _postprocess_addresses(
    addresses: List[MergeAddressOutput],
    slot_to_url: Dict[int, str],
) -> List[AddressOutput]:
    deduped: List[AddressOutput] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for item in addresses:
        source_url = slot_to_url.get(item.sourceSlot)
        if source_url is None:
            continue

        dedupe_key = (
            _normalize_for_dedupe(item.description),
            _normalize_for_dedupe(item.address),
            source_url,
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        deduped.append(
            AddressOutput(
                description=item.description.strip(),
                address=item.address.strip(),
                sourceUrl=source_url,
            )
        )

    deduped.sort(key=lambda item: 0 if _is_hq(item.description) else 1)
    return deduped[:5]
