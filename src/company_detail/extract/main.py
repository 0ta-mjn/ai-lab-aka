from langfuse import observe

from ..discover import CandidateUrl
from .schema import PageExtractionResult


@observe(
    as_type="span",
    name="extract_company_detail_from_page",
    capture_input=True,
    capture_output=True,
)
def extract_company_detail_from_page(
    candidate: CandidateUrl,
) -> PageExtractionResult:
    """
    フロー2: 抽出 (Extract)

    特定された候補URLから、事業内容と住所候補を抽出する。

    Args:
        candidate (CandidateUrl): フロー1で見つかった候補URL(1件)

    Returns:
        PageExtractionResult: 1ページ分の抽出結果
    """
    # TODO: 実装
    # 1. Jina AI Readerでページ取得
    # 2. LLMで情報抽出
    raise NotImplementedError
