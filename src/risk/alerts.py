from __future__ import annotations


def assign_alert_level(score: float, thresholds: dict[str, float]) -> str:
    if score >= thresholds["high"]:
        return "HIGH"
    if score >= thresholds["medium"]:
        return "MEDIUM"
    return "LOW"


def apply_hysteresis(
    previous_level: str,
    score: float,
    thresholds: dict[str, float],
    margin: float = 0.05,
) -> str:
    raw_level = assign_alert_level(score, thresholds)

    if previous_level == "HIGH":
        if score >= thresholds["high"] - margin:
            return "HIGH"
    if previous_level == "MEDIUM":
        if raw_level == "LOW" and score >= thresholds["medium"] - margin:
            return "MEDIUM"
    return raw_level
