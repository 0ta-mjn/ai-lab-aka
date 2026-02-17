from typing import List

from pydantic import BaseModel


class AddressItem(BaseModel):
    description: str
    address: str


class ExtractedContent(BaseModel):
    business: List[str]
    addresses: List[AddressItem]


class PageExtractionResult(BaseModel):
    url: str
    title: str
    extracted: ExtractedContent
