from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from src.api.schemas import AlertCreate, AlertListResponse, AlertRecord


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            camera_id TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            score REAL NOT NULL,
            evidence_start INTEGER NOT NULL,
            evidence_end INTEGER NOT NULL
        )
        """
    )
    conn.commit()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


def create_app(db_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Crowd Alert Service", version="0.1.0")
    resolved_db = db_path or Path("artifacts/alerts.db")
    app.state.db = _connect(resolved_db)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict[str, Any]:
        row = app.state.db.execute("SELECT COUNT(*) AS c FROM alerts").fetchone()
        return {"service": "up", "alerts_count": int(row["c"])}

    @app.post("/alerts", status_code=201)
    def create_alert(payload: AlertCreate) -> AlertRecord:
        cursor = app.state.db.execute(
            """
            INSERT INTO alerts (timestamp, camera_id, risk_level, score, evidence_start, evidence_end)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload.timestamp,
                payload.camera_id,
                payload.risk_level,
                payload.score,
                payload.evidence_window[0],
                payload.evidence_window[1],
            ),
        )
        app.state.db.commit()
        alert_id = int(cursor.lastrowid)
        return AlertRecord(id=alert_id, **payload.model_dump())

    @app.get("/alerts")
    def list_alerts(limit: int = 50) -> AlertListResponse:
        rows = app.state.db.execute(
            """
            SELECT id, timestamp, camera_id, risk_level, score, evidence_start, evidence_end
            FROM alerts
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        alerts = [
            AlertRecord(
                id=int(r["id"]),
                timestamp=float(r["timestamp"]),
                camera_id=str(r["camera_id"]),
                risk_level=str(r["risk_level"]),
                score=float(r["score"]),
                evidence_window=(int(r["evidence_start"]), int(r["evidence_end"])),
            )
            for r in rows
        ]
        return AlertListResponse(alerts=alerts)

    return app


app = create_app()
