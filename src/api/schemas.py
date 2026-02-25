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
    acknowledged_by: str | None = None
    ack_note: str | None = None
    acknowledged_at: float | None = None


class AlertListResponse(BaseModel):
    alerts: list[AlertRecord]


class AlertAckRequest(BaseModel):
    operator_name: str
    note: str


class ThresholdProfile(BaseModel):
    profile_name: str = "default"
    low: float = Field(ge=0.0, le=1.0)
    medium: float = Field(ge=0.0, le=1.0)
    high: float = Field(ge=0.0, le=1.0)


class TimelinePoint(BaseModel):
    timestamp: float
    score: float
    risk_level: RiskLevel
    camera_id: str


class DashboardSummaryResponse(BaseModel):
    alerts: list[AlertRecord]
    timeline: list[TimelinePoint]
    level_counts: dict[str, int]
