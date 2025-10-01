
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DonorDoc(BaseModel):
    type: str = "donor_profile"
    donor_id: int
    display_name: str
    city: str
    state: str
    capacity_estimate: int
    affinity_score: float
    gifts_3yr: float
    last_gift_dt: Optional[str] = None
    last_touch_dt: Optional[str] = None
    interests: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    donor_id: int
    display_name: str
    city: str
    state: str
    capacity_estimate: int
    affinity_score: float
    gifts_3yr: float
    last_touch_dt: Optional[str]
    score: float
    provenance: Dict[str, Any]


class SearchFilters(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_bucket: Optional[List[int]] = None
    interests: Optional[List[str]] = None


class SearchRequest(BaseModel):
    query: str
    filters: SearchFilters = Field(default_factory=SearchFilters)
    k: int = 30


class ItineraryMeeting(BaseModel):
    time: str
    donor_id: int
    display_name: str
    rationale: str
    score_breakdown: Dict[str, Any]
    provenance: Dict[str, Any]


class DayPlan(BaseModel):
    date: str
    city: str
    meetings: List[ItineraryMeeting]


class ItineraryResponse(BaseModel):
    itinerary: List[DayPlan]
    coverage_notes: str | None = None
    data_gaps: List[str] = Field(default_factory=list)
