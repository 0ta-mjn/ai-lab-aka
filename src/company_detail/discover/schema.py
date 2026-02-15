from typing import List

from pydantic import BaseModel


class CandidateUrl(BaseModel):
    url: str
    category: str
    reason: str


class DiscoveryResult(BaseModel):
    candidates: List[CandidateUrl]
