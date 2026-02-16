import logging
from typing import List

from langfuse import LangfuseSpan
from pydantic import BaseModel, Field

from src.infra.jina_ai import JinaReaderResponse, LinkItem, fetch_jina_reader_page
from src.infra.llm.generate_structured_output import generate_structured_output

from .schema import HubPageLinks
from .utils import is_same_domain

logger = logging.getLogger(__name__)


class HubSelectionResult(BaseModel):
    selected_indices: List[int] = Field(
        ...,
        description="List of indices of selected hub URLs",
    )


def explore_hubs(
    company_name: str, company_url: str, *, parent_span: LangfuseSpan | None = None
) -> List[HubPageLinks]:
    """
    Explore Hub pages and collect potential URLs grouped by hub pages.

    Returns a list of HubPageLinks:
    - title: page title
    - url: hub page url (includes top page)
    - links: same-domain links found on that hub page
    """

    # Fetch Top Page
    try:
        top_result = fetch_jina_reader_page(company_url)
    except Exception as e:
        logger.warning(f"Failed to fetch top page {company_url}: {e}")
        top_result = None

    if not top_result:
        logger.warning("Top page fetch failed. Returning empty list.")
        return []

    # Top hub item
    norm_top_url = top_result.url
    top_title = top_result.title or "Top Page"
    top_links = links_from_jina_response(norm_top_url, top_result)

    # Prepare list for LLM selection
    current_links_list = top_links

    # Limit for context
    limited_pool = current_links_list[:200]

    # Format for Prompt: "Index. [Title] (URL)"
    links_text = "\n".join(
        [f"{i}. [{link.title}] ({link.url})" for i, link in enumerate(limited_pool)]
    )

    hub_prompt = f"""
Target Company:
- name: {company_name}
- official_site: {company_url}

Index Range:
- valid_indices: 0..{len(limited_pool) - 1}
- select_count: 0..3

Available Links (index is global):
{links_text}
"""

    hub_indices = []
    try:
        hub_result = generate_structured_output(
            model="gemini/gemini-2.5-flash-lite",
            system_prompt="""
Role: Select hub-page candidates from a same-domain link list for a company website discovery workflow.

Definition:
- A hub page is a navigational/category page that links to content pages where we can later extract:
    - company profile (会社概要/企業情報/About)
    - business/services (事業内容/サービス/プロダクト)
    - locations/access (アクセス/所在地/拠点)

Selection rubric (priority order):
1) Pages clearly about company info/services/offices AND likely to contain many internal links
2) Top-level category pages (e.g., 会社情報, サービス, 拠点一覧)
3) Avoid low-signal or single-purpose pages: privacy/terms, news, blog, campaigns, IR, standalone articles

Hard constraints:
- Choose ONLY from the provided indices
- Select 0 to 4 items
- Return an empty list if none fit

Output:
- Return ONLY a JSON object that matches the output schema
- No explanations, no markdown, no extra keys
""",
            prompt=hub_prompt,
            output_schema=HubSelectionResult,
            generation_name="discover_select_hubs",
            metadata={
                "company_name": company_name,
                "company_url": company_url,
                "max_candidates": 5,
            },
            parent_span=parent_span,
        )
        hub_indices = hub_result.selected_indices
    except Exception as e:
        logger.error(f"Hub selection failed: {e}")

    # Fetch Hub Pages & Collect More Links
    hub_items: List[HubPageLinks] = [
        HubPageLinks(title=top_title, url=norm_top_url, links=top_links)
    ]
    for idx in hub_indices:
        if idx < 0 or idx >= len(limited_pool):
            continue

        hub_meta = limited_pool[idx]
        hub_url = hub_meta.url

        # Avoid re-fetching top page
        if hub_url == norm_top_url:
            continue

        try:
            hub_res = fetch_jina_reader_page(hub_url)
            if not hub_res:
                continue
            hub_title = (hub_res.title or hub_meta.title or hub_url).strip()
            hub_links = links_from_jina_response(hub_url, hub_res)
            hub_items.append(
                HubPageLinks(title=hub_title, url=hub_url, links=hub_links)
            )
        except Exception as e:
            logger.warning(f"Failed to fetch hub page {hub_url}: {e}")
            # Continue

    return hub_items


def links_from_jina_response(
    base_url: str, jina_result: JinaReaderResponse
) -> List[LinkItem]:
    """Extract same-domain links, ensuring uniqueness and normalization."""
    if not jina_result or not jina_result.links:
        return []

    discovered: dict[str, str] = {}
    for link_item in jina_result.links:
        norm_url = link_item.url
        if not is_same_domain(norm_url, base_url):
            continue

        if norm_url not in discovered:
            discovered[norm_url] = link_item.title
        elif len(link_item.title) > len(discovered[norm_url]):
            discovered[norm_url] = link_item.title

    links = [LinkItem(url=u, title=t) for u, t in discovered.items()]
    links.sort(key=lambda x: x.url)
    return links
