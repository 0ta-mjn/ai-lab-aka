from typing import Dict, List

from pydantic import BaseModel


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


class CompanyDetailOutput(BaseModel):
    """最終出力フォーマット"""

    company_name: str
    company_url: str
    address: List[AddressOutput]
    business_summary: BusinessSummaryOutput
    viewed_source_urls: List[str]
