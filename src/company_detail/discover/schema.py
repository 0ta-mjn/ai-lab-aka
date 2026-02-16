from typing import List

from pydantic import BaseModel

from src.infra.jina_ai import LinkItem


class CandidateUrl(BaseModel):
    url: str
    category: str
    reason: str


class DiscoveryResult(BaseModel):
    candidates: List[CandidateUrl]


class HubPageLinks(BaseModel):
    title: str
    url: str
    links: List[LinkItem]
