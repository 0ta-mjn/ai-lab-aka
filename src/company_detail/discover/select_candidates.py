import logging
from typing import List

from langfuse import LangfuseSpan
from pydantic import BaseModel, Field

from src.infra.llm.generate_structured_output import generate_structured_output

from .schema import (
    CandidateUrl,
    DiscoveryResult,
    HubPageLinks,
)
from .utils import is_same_domain

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


def select_candidates(
    company_name: str,
    company_url: str,
    available_hubs: List[HubPageLinks],
    *,
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
            model="gemini/gemini-2.5-flash-lite",
            system_prompt="""
Role: Select the best candidate pages for downstream extraction from a provided same-domain link list.

Downstream use:
- Each selected page will be fetched and an extraction step will try to pull:
    - addresses (本社/拠点/所在地)
    - business/service facts (事業内容/サービス/プロダクト)
- Prefer pages that likely CONTAIN the information (content pages), not just navigation link lists.

Selection rubric (aim for balance):
- Address-focused pages: 1-2
    - Examples: 会社概要 with 所在地, アクセス, 拠点一覧, 会社情報 where address is written
- Business-focused pages: 1-3
    - Examples: 事業内容, サービス一覧, プロダクト/ソリューション
- If available, include a company profile/about page (会社概要/企業情報) because it often contains the official address.

Avoid selecting (unless there is no better option):
- プライバシーポリシー/利用規約/免責
- ニュース/プレスリリース/ブログ/イベント/キャンペーン
- IR/投資家情報（住所が載る場合もあるが優先度は低い）
- 問い合わせフォームのみのページ

Rules:
- Choose ONLY from the provided indices
- Do not select near-duplicates (language duplicates or tracking variants)

For each selection, provide:
- index: the chosen index
- category: a short snake_case label (free text)
- reason (Japanese): 1-2 sentences; explicitly state whether it likely contains "住所" and/or "事業内容" and why

List format note:
- The list may be grouped with Markdown headers like "# ..." for readability
- Indices are global across the entire list (not per section)

Output:
- Return ONLY a JSON object that matches the output schema
- No prose, no markdown, no extra keys
""",
            prompt=selection_prompt,
            output_schema=CandidateSelectionResult,
            generation_name="discover_select_candidates",
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

        if is_same_domain(hub.url, company_url) and hub.url not in seen_urls:
            seen_urls.add(hub.url)
            pool_items.append((hub.url, hub_title, hub_title, hub.url))

        for link in hub.links:
            if not is_same_domain(link.url, company_url):
                continue

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
