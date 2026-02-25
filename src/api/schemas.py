from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class AlertCreate(BaseModel):
    timestamp: float
    camera_id: str
    risk_level: RiskLevel
    score: float = Field(ge=0.0, le=1.0)
    evidence_window: tuple[int, int]


class AlertRecord(AlertCreate):
    id: int


class AlertListResponse(BaseModel):
    alerts: list[AlertRecord]
