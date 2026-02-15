import logging
import os
from typing import Dict, Optional, TypedDict

import httpx
from langfuse import get_client, observe

# Configure logger
logger = logging.getLogger(__name__)


class JinaReaderResponse(TypedDict):
    content: str
    links: Dict[str, str]
    title: Optional[str]
    description: Optional[str]
    url: str


@observe(
    as_type="generation",
    name="fetch_jina_reader_page",
    capture_output=True,
)
def fetch_jina_reader_page(url: str) -> Optional[JinaReaderResponse]:
    """
    Jina AI Readerを使用してページを取得するラッパー関数。

    Args:
        url (str): 取得対象のURL

    Returns:
        Optional[JinaReaderResponse]: 取得成功時は抽出データを含む辞書、失敗時はNone

    Note:
        - 環境変数 JINA_AI_API_KEY が必要です。
        - 失敗時はログを出力し、Noneを返します
    """

    langfuse_context = get_client()

    jina_api_key = os.environ.get("JINA_AI_API_KEY")
    if not jina_api_key:
        raise ValueError("JINA_AI_API_KEY environment variable is not set.")

    langfuse_context.update_current_generation(
        model="jina-ai-reader",
        input={"url": url},
    )

    # Jina AI Reader endpoint
    target_url = f"https://r.jina.ai/{url}"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {jina_api_key}",
        "X-Locale": "ja-JP",
        "X-Retain-Images": "none",
        "X-With-Links-Summary": "true",
        "X-Base": "final",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(target_url, headers=headers)
            response.raise_for_status()

            response_json = response.json()
            data = response_json.get("data", {})

            # token使用量の取得 (data.usage.tokens または meta.usage.tokens)
            usage_tokens = data.get("usage", {}).get("tokens")
            if usage_tokens is None:
                usage_tokens = (
                    response_json.get("meta", {}).get("usage", {}).get("tokens", 0)
                )

            if usage_tokens:
                langfuse_context.update_current_generation(
                    usage_details={"input_tokens": 0, "output_tokens": usage_tokens}
                )

            result: JinaReaderResponse = {
                "content": data.get("content", ""),
                "links": data.get("links", {}),
                "title": data.get("title"),
                "description": data.get("description"),
                "url": data.get("url", url),
            }
            return result

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching {url} via Jina: {e}")
        langfuse_context.update_current_generation(
            status_message=f"HTTP error: {e.response.status_code}"
        )
        return None
