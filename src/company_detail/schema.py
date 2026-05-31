import re
from typing import Dict, List

from pydantic import BaseModel, field_validator


class CompanyDetailWorkflowInput(BaseModel):
    company_name: str
    company_url: str


class AddressOutput(BaseModel):
    description: str
    address: str
    sourceUrl: str


class BusinessSummaryOutput(BaseModel):
    detail: str
    sourceUrls: Dict[str, str]

    @field_validator("sourceUrls", mode="before")
    @classmethod
    def normalize_citation_keys(cls, value):
        if not isinstance(value, dict):
            return value

        normalized = {}
        for key, url in value.items():
            key_text = str(key).strip()
            match = re.fullmatch(r"\[?(\d+)\]?", key_text)
            normalized_key = match.group(1) if match else key_text
            normalized[normalized_key] = url
        return normalized


class CompanyDetailOutput(BaseModel):
    """最終出力フォーマット"""

    company_name: str
    company_url: str
    address: List[AddressOutput]
    business_summary: BusinessSummaryOutput
    viewed_source_urls: List[str]
