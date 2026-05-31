import re

from agents import function_tool
from pydantic import BaseModel

from src.company_detail.schema import (
    AddressOutput,
    BusinessSummaryOutput,
    CompanyDetailOutput,
)
from src.infra.jina_ai import fetch_jina_reader_page
from src.infra.langfuse.with_span import WithSpanContext
from src.infra.llm.agents import ToolExecution, run_agent_sync


class AgentCompanyDetailOutput(BaseModel):
    company_name: str
    company_url: str
    address: list[AddressOutput]
    business_summary: BusinessSummaryOutput


@function_tool
def access_url(url: str) -> str:
    """Fetch the text content and links of a web page by URL."""
    try:
        result = fetch_jina_reader_page(url)
        if result and result.content:
            links_summary = "\n".join(
                [f"- {link.title}: {link.url}" for link in result.links[:200]]
            )
            return (
                f"# Page URL\n{result.url}\n\n"
                f"# Page Content\n{result.content}\n\n"
                f"# Found Links\n{links_summary}"
            )
        return f"Error: Could not fetch page content from {url}."
    except Exception as e:
        return f"Error fetching page {url}: {str(e)}"


def run_company_detail_agent(
    company_name: str,
    company_url: str,
    *,
    span_context: WithSpanContext | None = None,
) -> CompanyDetailOutput:
    """
    LLM Based Approach (Pattern 2):
    Uses a single LLM agent to freely explore the company website, fetch pages,
    and output the final structured schema.
    """
    system_prompt = """あなたは、企業公式サイトから会社情報を収集するAIエージェントです。
入力として企業名と公式サイトURLが与えられます。

目的:
- 企業の住所情報と事業概要を、公式サイト上の根拠に基づいて抽出してください。
- workflow版と同じ要件で最終JSONを作成してください。

探索方針:
1. 最初に入力された公式サイトURLを `access_url` ツールで取得してください。
2. ページ本文と `Found Links` を読み、会社概要、企業情報、事業内容、サービス、プロダクト、アクセス、所在地、拠点一覧に関係するページを優先して追加取得してください。
3. プライバシーポリシー、利用規約、ニュース、ブログ、イベント、キャンペーン、問い合わせフォームのみのページは、より適切なページがある場合は避けてください。
4. URLは初期入力URL、または `Found Links` に出てきたURLだけを使用してください。URLを推測して作らないでください。

最終出力要件:
- JSONスキーマに一致するJSONだけを返してください。説明文やMarkdownは不要です。
- `company_name` と `company_url` は入力値に合わせてください。
- `address` は最大5件です。本社・本店・本社オフィスを優先し、その後に主要拠点を入れてください。
- 各住所は `description`、ページに書かれた住所文字列そのものに近い `address`、根拠ページの `sourceUrl` を入れてください。
- 住所に建物名、ビル名、施設名、階数、部屋番号、郵便番号が含まれる場合は省略しないでください。番地だけで止めず、同じ住所欄に書かれている末尾情報まで保持してください。
- 住所は、所在地、本社、支社、営業所、アクセス、住所などの文脈が明確なものだけを採用してください。電話番号、FAX、メールアドレスだけの行は住所として扱わないでください。
- `business_summary.detail` は日本語で、公式サイトに書かれた事業・サービス・プロダクトの事実を要約してください。
- 事業概要には根拠番号 `[1]`, `[2]` のような引用番号を含めてください。
- `business_summary.sourceUrls` は、本文中の引用番号と根拠URLの対応だけを入れてください。keyは `"1"` のような数字文字列にし、`"[1]"` のように角括弧を含めないでください。

禁止事項:
- ページ本文にない住所・事業内容を推測しないでください。
- 根拠として閲覧していないURLを `sourceUrl` や `business_summary.sourceUrls` に入れないでください。
- 採用する情報は公式サイト上の内容に限定してください。
"""

    input_text = f"以下の企業情報を収集してください。\n企業名: {company_name}\n公式サイトURL: {company_url}"

    agent_result = run_agent_sync(
        model="openai/gpt-5.4-mini",
        system_prompt=system_prompt,
        prompt=input_text,
        tools=[access_url],
        output_schema=AgentCompanyDetailOutput,
        generation_name="run_company_detail_agent",
        span_context=span_context,
    )

    output = agent_result.final_output
    return CompanyDetailOutput(
        company_name=output.company_name,
        company_url=output.company_url,
        address=output.address,
        business_summary=output.business_summary,
        viewed_source_urls=_extract_accessed_urls(agent_result.tool_executions),
    )


def _extract_accessed_urls(tool_executions: list[ToolExecution]) -> list[str]:
    urls = []
    seen = set()

    for execution in tool_executions:
        if execution.tool_name != "access_url":
            continue

        url = _extract_page_url_from_tool_output(execution.output)
        if url is None:
            url = _extract_requested_url(execution.arguments)
        if url is None:
            continue

        normalized = _normalize_url_for_dedupe(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        urls.append(url)

    return urls


def _extract_requested_url(arguments) -> str | None:
    if isinstance(arguments, dict):
        url = arguments.get("url")
        return url if isinstance(url, str) else None
    return None


def _extract_page_url_from_tool_output(output) -> str | None:
    if not isinstance(output, str):
        return None
    match = re.search(r"^# Page URL\s*\n(.+)$", output, flags=re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def _normalize_url_for_dedupe(url: str) -> str:
    return url.strip().rstrip("/")
