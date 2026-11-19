from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ArtifactCheckResult(BaseModel):
    module: str
    status: str
    score: Optional[float] = None
    is_suspicious: Optional[bool] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class DocumentAuthSummary(BaseModel):
    request_id: Optional[str] = None
    status: str = "completed"
    overall_risk_score: Optional[float] = None
    decision: str = "inconclusive"
    checks: List[ArtifactCheckResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)