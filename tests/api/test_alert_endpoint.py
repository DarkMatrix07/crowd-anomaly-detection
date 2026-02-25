from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app


def test_alert_post_and_get_persists_records(tmp_path: Path) -> None:
    db_path = tmp_path / "alerts.db"
    app = create_app(db_path=db_path)
    client = TestClient(app)

    payload = {
        "timestamp": 1700000000.0,
        "camera_id": "cam-1",
        "risk_level": "HIGH",
        "score": 0.91,
        "evidence_window": [10, 20],
    }

    post_resp = client.post("/alerts", json=payload)
    assert post_resp.status_code == 201

    list_resp = client.get("/alerts")
    assert list_resp.status_code == 200
    body = list_resp.json()
    assert len(body["alerts"]) == 1
    assert body["alerts"][0]["camera_id"] == "cam-1"
    assert body["alerts"][0]["risk_level"] == "HIGH"


def test_health_endpoint() -> None:
    app = create_app()
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
