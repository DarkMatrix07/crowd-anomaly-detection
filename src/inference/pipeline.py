from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np

from src.features.crowd_metrics import build_feature_row
from src.features.optical_flow import compute_optical_flow
from src.risk.alerts import apply_hysteresis
from src.risk.score import compute_risk_score


class RollingInferencePipeline:
    def __init__(
        self,
        camera_id: str,
        clip_length: int,
        clip_stride: int,
        anomaly_model_fn: Callable[[np.ndarray], float],
        thresholds: dict[str, float] | None = None,
        smoothing_alpha: float = 0.35,
        hysteresis_margin: float = 0.05,
    ) -> None:
        if clip_length <= 0 or clip_stride <= 0:
            raise ValueError("clip_length and clip_stride must be > 0")
        self.camera_id = camera_id
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.anomaly_model_fn = anomaly_model_fn
        self.thresholds = thresholds or {"low": 0.3, "medium": 0.6, "high": 0.85}
        self.smoothing_alpha = smoothing_alpha
        self.hysteresis_margin = hysteresis_margin

        self._buffer: deque[np.ndarray] = deque(maxlen=clip_length)
        self._frame_idx = -1
        self._last_emit_start = -clip_stride
        self._prev_level = "LOW"
        self._prev_smoothed_score = 0.0
        self._last_raw_score = 0.0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> list[dict[str, object]]:
        self._frame_idx += 1
        self._buffer.append(frame)

        if len(self._buffer) < self.clip_length:
            return []

        clip_start = self._frame_idx - self.clip_length + 1
        if clip_start - self._last_emit_start < self.clip_stride:
            return []
        self._last_emit_start = clip_start

        clip = np.stack(self._buffer, axis=0)
        anomaly_score = float(np.clip(self.anomaly_model_fn(clip), 0.0, 1.0))
        flow = compute_optical_flow(clip, method="farneback")
        feature_row = build_feature_row(clip, flow)

        trend_acceleration = max(0.0, anomaly_score - self._last_raw_score)
        raw_risk = compute_risk_score(
            anomaly_score=anomaly_score,
            flow_instability=float(feature_row["flow_variance_magnitude"]),
            density_pressure=float(feature_row["density_pressure"]),
            trend_acceleration=float(trend_acceleration),
        )
        smoothed_risk = (
            self.smoothing_alpha * raw_risk + (1.0 - self.smoothing_alpha) * self._prev_smoothed_score
        )
        level = apply_hysteresis(
            previous_level=self._prev_level,
            score=smoothed_risk,
            thresholds=self.thresholds,
            margin=self.hysteresis_margin,
        )

        self._prev_level = level
        self._prev_smoothed_score = smoothed_risk
        self._last_raw_score = raw_risk

        event = {
            "timestamp": float(timestamp),
            "camera_id": self.camera_id,
            "risk_level": level,
            "score": float(np.clip(smoothed_risk, 0.0, 1.0)),
            "evidence_window": [int(clip_start), int(self._frame_idx)],
        }
        return [event]
