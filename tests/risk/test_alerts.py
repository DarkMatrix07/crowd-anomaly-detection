from src.risk.alerts import apply_hysteresis, assign_alert_level


def test_assign_alert_level_uses_thresholds() -> None:
    thresholds = {"low": 0.3, "medium": 0.6, "high": 0.85}

    assert assign_alert_level(0.2, thresholds) == "LOW"
    assert assign_alert_level(0.61, thresholds) == "MEDIUM"
    assert assign_alert_level(0.9, thresholds) == "HIGH"


def test_hysteresis_reduces_level_flicker() -> None:
    thresholds = {"low": 0.3, "medium": 0.6, "high": 0.85}
    # Raw mapping would drop to LOW, but hysteresis keeps MEDIUM near boundary.
    level = apply_hysteresis(previous_level="MEDIUM", score=0.58, thresholds=thresholds, margin=0.05)

    assert level == "MEDIUM"
