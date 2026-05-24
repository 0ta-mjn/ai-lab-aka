from agents import function_tool

from src.company_detail.schema import CompanyDetailOutput
from src.infra.jina_ai import fetch_jina_reader_page
from src.infra.langfuse.with_span import WithSpanContext
from src.infra.llm.agents import run_agent_sync


@function_tool
def access_url(url: str) -> str:
    """Fetch the text content and links of a web page by URL."""
    try:
        result = fetch_jina_reader_page(url)
        if result and result.content:
            links_summary = "\n".join(
                [f"- {link.title}: {link.url}" for link in result.links[:30]]
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
    system_prompt = """You are an autonomous AI agent designed to collect detailed company information (addresses and business summary) by navigating their official website.
You will be provided with the company's name and official URL.

Your process:
1. Navigate to the company's official website using the `access_url` tool.
2. Read the page content and look at the "Found Links" section to find pages like '会社概要' (Company Profile), '事業内容' (Business Summary), or 'アクセス' (Access).
3. Call `access_url` on these newly found URLs to gather more specific information if necessary.
4. Once you have gathered sufficient information (up to 5 addresses prioritizing the head office, and a detailed business summary), output the final result.

Constraints & Rules:
- Do not make up or hallucinate URLs. Only use URLs found in the "Found Links" section or the initial input.
- `address` array: Output up to 5 addresses. Provide the description (e.g. "本社", "東京支店"), the raw address, and the `sourceUrl` where you found it.
- `business_summary.detail`: Write a comprehensive summary in Japanese based on the extracted business contents. Include citation numbers like [1], [2].
- `business_summary.sourceUrls`: Map the citation numbers (as strings, e.g., "1", "2") to the source URLs you viewed.
- `viewed_source_urls`: List all the URLs you successfully accessed using the `access_url` tool during this process.
"""

    input_text = f"Please collect company details for:\nCompany Name: {company_name}\nCompany URL: {company_url}"

    return run_agent_sync(
        model="openai/gpt-5.4-mini",
        system_prompt=system_prompt,
        prompt=input_text,
        tools=[access_url],
        output_schema=CompanyDetailOutput,
        generation_name="run_company_detail_agent",
        span_context=span_context,
    )
