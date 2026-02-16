import logging

from src.infra.langfuse import WithSpanContext, with_langfuse_span

from .explore_hubs import explore_hubs
from .schema import DiscoveryResult
from .select_candidates import select_candidates

logger = logging.getLogger(__name__)


def discover_company_detail_candidates(
    company_name: str,
    company_url: str,
    *,
    span_context: WithSpanContext | None = None,
) -> DiscoveryResult:
    """
    フロー1: ページ探索 (Discover)

    公式サイト内から最大5URLまで、企業概要・事業内容・住所を取得できそうなページを特定する。

    Args:
        company_name (str): 企業名
        company_url (str): 企業公式サイトURL

    Returns:
        DiscoveryResult: 候補URLのリストを含むオブジェクト
    """

    with with_langfuse_span(
        span_name="discover_pages",
        span_context=span_context,
    ) as span:
        span.set_input({"company_name": company_name, "company_url": company_url})

        # Explore Hubs
        hubs = explore_hubs(company_name, company_url, parent_span=span.span)

        # Select Candidates
        discovery_result = select_candidates(
            company_name,
            company_url,
            hubs,
            parent_span=span.span,
        )

        return span.finish(discovery_result)
