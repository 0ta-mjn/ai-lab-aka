from typing import List

from pydantic import BaseModel


class BusinessFact(BaseModel):
    text: str
    evidence: List[str]
    confidence: float


class AddressExtraction(BaseModel):
    description: str
    address: str
    evidence: List[str]
    confidence: float
    type: str  # headquarters | branch | unknown


class ExtractedContent(BaseModel):
    business_facts: List[BusinessFact]
    addresses: List[AddressExtraction]


class PageExtractionResult(BaseModel):
    url: str
    extracted: ExtractedContent
