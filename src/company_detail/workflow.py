from src.infra.langfuse import WithSpanContext, with_langfuse_span

from .discover import discover_company_detail_candidates
from .extract import extract_company_detail_from_page
from .merge import merge_company_detail_extractions
from .schema import CompanyDetailOutput


def run_company_detail_workflow(
    company_name: str,
    company_url: str,
    *,
    span_context: WithSpanContext | None = None,
) -> CompanyDetailOutput:
    """
    Company Detail Workflow メインエントリポイント (Orchestrator)

    企業名とURLを入力として、事業内容と住所を抽出・統合して返す。
    Langfuse Traceとして1回の実行を記録する。

    Args:
        company_name (str): 企業名
        company_url (str): 企業URL

    Returns:
        CompanyDetailOutput: 最終的な抽出結果
    """

    with with_langfuse_span(
        span_name="run_company_detail_workflow",
        span_context=span_context,
    ) as obs:
        try:
            obs.set_input({"company_name": company_name, "company_url": company_url})

            # 1. Page Discovery (探索)
            discovery_result = discover_company_detail_candidates(
                company_name, company_url
            )

            # 2. Extraction (抽出)
            extraction_results = [
                extract_company_detail_from_page(candidate)
                for candidate in discovery_result.candidates
            ]

            # 3. Merge (統合)
            final_output = merge_company_detail_extractions(extraction_results)
            return obs.finish(final_output)
        except Exception as e:
            obs.error(e)
            raise
