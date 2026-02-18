import json
import logging
import re
import unicodedata
from typing import Dict, List
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from src.infra.langfuse import WithSpanContext
from src.infra.llm import generate_structured_output

from ..extract import PageExtractionResult
from ..schema import AddressOutput, BusinessSummaryOutput, CompanyDetailOutput

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
        model="openai/gpt-5-mini",
        system_prompt="""
Role:
- Merge page-level extraction results and produce final structured output.

Non-negotiable rules:
- Return only JSON matching schema. No markdown, no prose, no extra keys.
- Never output URL strings. Use only sourceSlot/citationSlots as slot references.
- Use only provided page_extractions as evidence.

Address rules:
- Include extracted addresses with sourceSlot.
- Put head office entries first when description indicates 本社.
- Output at most 5 addresses.

Business summary rules:
- Write business_summary.detail in Japanese and include citations like [1], [2].
- citationSlots must map each citation number string to sourceSlot integer.
- If no valid business evidence, return detail as empty string and citationSlots as [].

Fallback rules:
- If address evidence is missing, return address as [].

Output examples:
- addresses example:
    {
        "address": [
            {"description": "本社", "address": "東京都千代田区...", "sourceSlot": 1},
            {"description": "支社", "address": "大阪府大阪市...", "sourceSlot": 2}
        ]
    }
- business_summary example:
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
        generation_name="merge_and_format",
        metadata={
            "num_pages_used": len(extractions),
            "num_address_candidates": sum(
                len(item.extracted.addresses) for item in extractions
            ),
            "num_business_candidates": sum(
                len(item.extracted.business) for item in extractions
            ),
        },
        parent_span=span_context.get("parent_span") if span_context else None,
    )

    final_addresses = _postprocess_addresses(merged.address, slot_to_url)
    final_business_summary = _build_business_summary(
        merged.business_summary.detail,
        merged.business_summary.citationSlots,
        slot_to_url,
    )

    return CompanyDetailOutput(
        company_name=company_name,
        company_url=company_url,
        address=final_addresses,
        business_summary=final_business_summary,
        viewed_source_urls=list(slot_to_url.values()),
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
        key: valid_source_urls[key] for key in sorted(valid_source_urls.keys(), key=int)
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
