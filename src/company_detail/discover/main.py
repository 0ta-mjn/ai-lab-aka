from langfuse import observe

from .schema import DiscoveryResult


@observe(
    as_type="span",
    name="discover_company_detail_candidates",
    capture_input=True,
    capture_output=True,
)
def discover_company_detail_candidates(
    company_name: str, company_url: str
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
    # TODO: 実装
    # 1. ハブページ調査
    # 2. サイトマップ確認
    # 3. 候補選定
    raise NotImplementedError
