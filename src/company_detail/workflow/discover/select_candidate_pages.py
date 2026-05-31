import logging
from typing import List

from langfuse import LangfuseSpan
from pydantic import BaseModel, Field

from src.infra.llm.generate_structured_output import generate_structured_output
from src.infra.llm.registry import ModelName

from .schema import (
    CandidateUrl,
    DiscoveryResult,
    HubPageLinks,
)

logger = logging.getLogger(__name__)


class CandidateSelection(BaseModel):
    index: int = Field(
        ...,
        description="Index of the selected link from the provided list",
    )
    category: str = Field(..., description="Category of the page")
    reason: str = Field(..., description="Reason for selecting this URL")


class CandidateSelectionResult(BaseModel):
    selections: List[CandidateSelection]


def select_candidate_pages(
    company_name: str,
    company_url: str,
    available_hubs: List[HubPageLinks],
    *,
    model: ModelName = "gemini/gemini-3.1-flash-lite",
    parent_span: LangfuseSpan | None = None,
) -> DiscoveryResult:
    """
    Select up to 5 best candidate URLs for extraction.
    """

    # Keep the prompt bounded to avoid oversized context.
    max_pool_for_prompt = 200

    pool_items = _collect_unique_same_domain_pool_items(company_url, available_hubs)
    pool_items = _order_and_trim_pool_items(
        pool_items,
        max_pool_for_prompt=max_pool_for_prompt,
    )

    # Fallback for empty pool
    if not pool_items:
        return DiscoveryResult(candidates=[])

    # Format for Prompt:
    # # Hub/Page Title 1
    # 0. [title](https://example.com/xxx)
    # ...
    # # Hub/Page Title 2
    # ... (index is global across the entire list, not per section)
    links_lines: list[str] = []
    current_hub_title: str | None = None
    for i, (url, title, hub_title, _) in enumerate(pool_items):
        if hub_title != current_hub_title:
            if links_lines:
                links_lines.append("")
            links_lines.append(f"# {hub_title}")
            current_hub_title = hub_title
        links_lines.append(f"{i}. [{title}]({url})")
    links_text = "\n".join(links_lines)

    selection_prompt = f"""
Target Company:
- name: {company_name}
- official_site_domain_root: {company_url}

Index Range:
- valid_indices: 0..{len(pool_items) - 1}
- select_count: 0..5

Available Links:
{links_text}
"""

    try:
        selection_result = generate_structured_output(
            model=model,
            system_prompt="""
役割:
- 提示されたリンク一覧から、住所情報と事業概要の抽出に使う候補ページを選んでください。

後続処理の目的:
- 選ばれた各ページを取得し、以下を抽出します。
  - 住所: 本社、本店、本社オフィス、支社、営業所、所在地、アクセス、拠点
  - 事業内容: 事業、サービス、プロダクト、ソリューション、提供価値
- 単なるナビゲーションページより、実際に情報が書かれていそうな本文ページを優先してください。

選定基準:
- 住所系ページを1〜2件選んでください。例: 会社概要、企業情報、アクセス、所在地、拠点一覧。
- 事業系ページを1〜3件選んでください。例: 事業内容、サービス一覧、プロダクト、ソリューション。
- 会社概要・企業情報ページがある場合は、公式住所を含むことが多いため優先してください。
- 全体で最大5件まで選んでください。

避けるページ:
- プライバシーポリシー、利用規約、免責
- ニュース、プレスリリース、ブログ、イベント、キャンペーン
- IR、投資家情報
- 問い合わせフォームのみのページ
- この企業の公式ページではないページ

制約:
- 必ず提示されたindexだけから選んでください。
- 類似ページ、言語違い、トラッキング違いなどの重複は避けてください。

各選択に含める情報:
- index: 選択したindex
- category: 短いsnake_caseラベル
- reason: 日本語で1〜2文。住所または事業内容を含みそうな理由を書いてください。

リスト形式の注意:
- リストはMarkdown見出しでグループ化されている場合があります。
- indexはセクションごとではなく、リスト全体で一意です。

出力:
- 出力スキーマに一致するJSONだけを返してください。
- 説明文、Markdown、余分なキーは不要です。
""",
            prompt=selection_prompt,
            output_schema=CandidateSelectionResult,
            generation_name="select_candidate_pages",
            metadata={"company_name": company_name, "company_url": company_url},
            parent_span=parent_span,
        )
    except Exception:
        logger.exception("Candidate selection failed")
        # Return empty
        empty_res = DiscoveryResult(candidates=[])
        return empty_res

    # Map back to CandidateUrl
    final_candidates: list[CandidateUrl] = []
    selected_urls: set[str] = set()
    for selection in selection_result.selections:
        idx = selection.index
        if idx < 0 or idx >= len(pool_items):
            continue

        url, _, _, _ = pool_items[idx]

        if url in selected_urls:
            continue
        selected_urls.add(url)

        final_candidates.append(
            CandidateUrl(
                url=url,
                category=selection.category,
                reason=selection.reason,
            )
        )

    discovery_result = DiscoveryResult(candidates=final_candidates)
    return discovery_result


def _collect_unique_same_domain_pool_items(
    company_url: str,
    available_hubs: List[HubPageLinks],
) -> list[tuple[str, str, str, str]]:
    def _normalize_title(title: str, fallback: str) -> str:
        t = (title or "").strip()
        return t if t else fallback

    # Collect unique, same-domain URLs from hub pages (including hub URLs themselves).
    # Keep the first-seen metadata for each URL.
    seen_urls: set[str] = set()
    pool_items: list[tuple[str, str, str, str]] = []

    for hub in available_hubs:
        hub_title = _normalize_title(hub.title, hub.url)

        seen_urls.add(hub.url)
        pool_items.append((hub.url, hub_title, hub_title, hub.url))

        for link in hub.links:
            title = _normalize_title(link.title, link.url)
            if link.url in seen_urls:
                continue

            seen_urls.add(link.url)
            pool_items.append((link.url, title, hub_title, hub.url))

    return pool_items


def _order_and_trim_pool_items(
    pool_items: list[tuple[str, str, str, str]],
    *,
    max_pool_for_prompt: int,
) -> list[tuple[str, str, str, str]]:
    # TODO: add heuristic ordering before LLM selection (title-based, url-based)
    # Deterministic ordering: group by source hub title in prompt for readability.
    pool_items.sort(key=lambda x: (x[2], x[0]))
    return pool_items[:max_pool_for_prompt]
