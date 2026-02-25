from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app


def _seed_alert(client: TestClient, score: float, level: str, ts: float) -> int:
    payload = {
        "timestamp": ts,
        "camera_id": "cam-1",
        "risk_level": level,
        "score": score,
        "evidence_window": [5, 12],
    }
    resp = client.post("/alerts", json=payload)
    assert resp.status_code == 201
    return int(resp.json()["id"])


def test_dashboard_summary_contract(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "alerts.db")
    client = TestClient(app)
    _seed_alert(client, score=0.22, level="LOW", ts=1700000001.0)
    _seed_alert(client, score=0.64, level="MEDIUM", ts=1700000002.0)
    _seed_alert(client, score=0.91, level="HIGH", ts=1700000003.0)

    resp = client.get("/dashboard/summary?limit=10")
    assert resp.status_code == 200
    body = resp.json()

    assert set(body) == {"alerts", "timeline", "level_counts"}
    assert len(body["alerts"]) == 3
    assert set(body["timeline"][0]) == {"timestamp", "score", "risk_level", "camera_id"}
    assert set(body["level_counts"]) == {"low", "medium", "high"}


def test_threshold_profile_roundtrip(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "alerts.db")
    client = TestClient(app)

    put_resp = client.put(
        "/config/thresholds",
        json={
            "profile_name": "strict",
            "low": 0.35,
            "medium": 0.65,
            "high": 0.9,
        },
    )
    assert put_resp.status_code == 200

    get_resp = client.get("/config/thresholds")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["profile_name"] == "strict"
    assert body["low"] == 0.35
    assert body["medium"] == 0.65
    assert body["high"] == 0.9


def test_alert_acknowledgment_contract(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "alerts.db")
    client = TestClient(app)
    alert_id = _seed_alert(client, score=0.91, level="HIGH", ts=1700000003.0)

    ack_resp = client.post(
        f"/alerts/{alert_id}/ack",
        json={"operator_name": "operator-1", "note": "Dispatched field team"},
    )
    assert ack_resp.status_code == 200

    list_resp = client.get("/alerts")
    assert list_resp.status_code == 200
    first = list_resp.json()["alerts"][0]
    assert first["acknowledged_by"] == "operator-1"
    assert first["ack_note"] == "Dispatched field team"
    assert first["acknowledged_at"] is not None
