from typing import List

from langfuse import observe

from ..extract import PageExtractionResult
from ..schema import CompanyDetailOutput


@observe(
    as_type="span",
    name="merge_company_detail_extractions",
    capture_input=True,
    capture_output=True,
)
def merge_company_detail_extractions(
    extractions: List[PageExtractionResult],
) -> CompanyDetailOutput:
    """
    フロー3: 統合 (Merge)

    複数のページからの抽出結果を統合し、最終的なJSON形式を生成する。
    - 同一ドメイン情報を優先
    - 本社を先頭に配置
    - 参照番号の割り当て

    Args:
        extractions (List[PageExtractionResult]): ページごとの抽出結果リスト

    Returns:
        CompanyDetailOutput: 統合された最終結果
    """
    # TODO: 実装
    # 1. 抽出結果の統合生成 (LLM)
    # 2. フォーマット整形
    raise NotImplementedError
