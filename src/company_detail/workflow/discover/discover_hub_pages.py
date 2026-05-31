import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from langfuse import LangfuseSpan
from pydantic import BaseModel, Field

from src.infra.jina_ai import JinaReaderResponse, LinkItem, fetch_jina_reader_page
from src.infra.llm.generate_structured_output import generate_structured_output
from src.infra.llm.registry import ModelName

from .schema import HubPageLinks

logger = logging.getLogger(__name__)


class HubSelectionResult(BaseModel):
    selected_indices: List[int] = Field(
        ...,
        description="List of indices of selected hub URLs",
    )


def discover_hub_pages(
    company_name: str,
    company_url: str,
    *,
    max_hub_candidates: int = 5,
    model: ModelName = "gemini/gemini-3.1-flash-lite",
    parent_span: LangfuseSpan | None = None,
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
        top_result = fetch_jina_reader_page(company_url, tool_name="fetch_initial_page")
    except Exception as e:
        logger.warning(f"Failed to fetch top page {company_url}: {e}")
        top_result = None

    if not top_result:
        logger.warning("Top page fetch failed. Returning empty list.")
        return []

    # Top hub item
    norm_top_url = top_result.url
    top_title = top_result.title or "Top Page"
    top_links = _links_from_jina_response(norm_top_url, top_result)

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
- select_count: 0..{max_hub_candidates}

Available Links (index is global):
{links_text}
"""

    hub_indices = []
    try:
        hub_result = generate_structured_output(
            model=model,
            system_prompt="""
役割:
- 企業公式サイトのリンク一覧から、後続の抽出に役立つハブページ候補を選んでください。

ハブページの定義:
- 会社概要、企業情報、事業内容、サービス、プロダクト、アクセス、所在地、拠点一覧などの詳細ページへ移動しやすい案内ページです。

選定基準:
1. 会社情報、事業・サービス、所在地・拠点に関係し、内部リンクを多く含みそうなページを優先してください。
2. 会社情報、サービス、事業紹介、拠点一覧などの上位カテゴリページを優先してください。
3. プライバシーポリシー、利用規約、ニュース、ブログ、イベント、キャンペーン、問い合わせフォームのみのページは避けてください。

制約:
- 必ず提示されたindexだけから選んでください。
- 0件から4件まで選んでください。
- 適切な候補がない場合は空配列を返してください。

出力:
- 出力スキーマに一致するJSONだけを返してください。
- 説明文、Markdown、余分なキーは不要です。
""",
            prompt=hub_prompt,
            output_schema=HubSelectionResult,
            generation_name="discover_hub_pages",
            metadata={
                "company_name": company_name,
                "company_url": company_url,
                "max_candidates": max_hub_candidates,
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
    hub_metas = []
    for idx in hub_indices:
        if idx < 0 or idx >= len(limited_pool):
            continue

        hub_meta = limited_pool[idx]
        hub_url = hub_meta.url

        # Avoid re-fetching top page
        if hub_url == norm_top_url:
            continue

        hub_metas.append(hub_meta)

    def fetch_hub_item(hub_meta: LinkItem) -> HubPageLinks | None:
        hub_url = hub_meta.url
        try:
            hub_res = fetch_jina_reader_page(hub_url, tool_name="fetch_hub_page")
            if not hub_res:
                return None
            hub_title = (hub_res.title or hub_meta.title or hub_url).strip()
            hub_links = _links_from_jina_response(hub_url, hub_res)
            return HubPageLinks(title=hub_title, url=hub_url, links=hub_links)
        except Exception as e:
            logger.warning(f"Failed to fetch hub page {hub_url}: {e}")
            return None

    if hub_metas:
        max_workers = min(max_hub_candidates, len(hub_metas))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for hub_item in executor.map(fetch_hub_item, hub_metas):
                if hub_item is None:
                    continue
                hub_items.append(hub_item)

    return hub_items


def _links_from_jina_response(
    base_url: str, jina_result: JinaReaderResponse
) -> List[LinkItem]:
    """Ensuring uniqueness and normalization."""
    if not jina_result or not jina_result.links:
        return []

    discovered: dict[str, str] = {}
    for link_item in jina_result.links:
        norm_url = link_item.url

        if _is_blacklisted(norm_url):
            continue

        if norm_url not in discovered:
            discovered[norm_url] = link_item.title
        elif len(link_item.title) > len(discovered[norm_url]):
            discovered[norm_url] = link_item.title

    links = [LinkItem(url=u, title=t) for u, t in discovered.items()]
    links.sort(key=lambda x: x.url)
    return links


def _is_blacklisted(url: str) -> bool:
    BLACK_LIST_DOMAIN = [
        "x.com",
        "twitter.com",
        "facebook.com",
        "instagram.com",
        "linkedin.com",
        "tiktok.com",
        "youtube.com",
        "line.me",
        "google.com/maps",
        "maps.google.com",
        "goo.gl/maps",
        # add more
    ]
    return any(domain in url for domain in BLACK_LIST_DOMAIN)
