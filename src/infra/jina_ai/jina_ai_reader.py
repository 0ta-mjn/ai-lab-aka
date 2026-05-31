import logging
import os
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from langfuse import get_client
from pydantic import BaseModel

from src.infra.langfuse.with_span import WithSpanContext

# Configure logger
logger = logging.getLogger(__name__)


def _normalize_url(base_url: str, link: str) -> str:
    """Resolve relative URLs and remove fragments."""
    try:
        absolute_url = urljoin(base_url, link)
        parsed = urlparse(absolute_url)
        if parsed.scheme not in ["http", "https"]:
            return link

        normalized = parsed.scheme + "://" + parsed.netloc + parsed.path
        if parsed.query:
            normalized += "?" + parsed.query
        return normalized
    except Exception:
        return link


class LinkItem(BaseModel):
    title: str
    url: str


class JinaReaderResponse(BaseModel):
    content: str
    links: List[LinkItem]
    title: Optional[str]
    description: Optional[str]
    url: str


def fetch_jina_reader_page(
    url: str, 
    *, 
    tool_name: str | None = None,
    span_context: WithSpanContext | None = None,
) -> Optional[JinaReaderResponse]:
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

    jina_api_key = os.environ.get("JINA_AI_API_KEY")
    if not jina_api_key:
        raise ValueError("JINA_AI_API_KEY environment variable is not set.")

    parent = span_context.get("parent_span") if span_context else get_client()

    with parent.start_as_current_observation(
        as_type="generation",
        name=tool_name or "fetch_jina_reader_page",
        model="jina-ai-reader",
        input={"url": url},
        level="DEBUG",
    ) as generation:
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
                    generation.update(
                        usage_details={"input_tokens": 0, "output_tokens": usage_tokens}
                    )

                # Jina API returns links as a dict {title: url}
                page_url = data.get("url", url)
                raw_links = data.get("links", {})
                formatted_links = []
                if isinstance(raw_links, dict):
                    for text, link_url in raw_links.items():
                        normalized_link_url = _normalize_url(page_url, link_url)
                        formatted_links.append(
                            LinkItem(
                                title=text.strip() if isinstance(text, str) else "",
                                url=normalized_link_url,
                            )
                        )

                result = JinaReaderResponse(
                    content=data.get("content", ""),
                    links=formatted_links,
                    title=data.get("title"),
                    description=data.get("description"),
                    url=page_url,
                )
                generation.update(output=result.model_dump())
                return result

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url} via Jina: {e}")
            generation.update(
                status_message=f"HTTP error: {e.response.status_code}",
                level="ERROR"
            )
            return None
        except Exception as e:
            generation.update(status_message=str(e), level="ERROR")
            raise
