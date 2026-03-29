from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    AlertAckRequest,
    AlertCreate,
    AlertListResponse,
    AlertRecord,
    DashboardSummaryResponse,
    ThresholdProfile,
    TimelinePoint,
)


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
            evidence_end INTEGER NOT NULL,
            acknowledged_by TEXT,
            ack_note TEXT,
            acknowledged_at REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS threshold_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            profile_name TEXT NOT NULL,
            low REAL NOT NULL,
            medium REAL NOT NULL,
            high REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO threshold_config (id, profile_name, low, medium, high, updated_at)
        SELECT 1, 'default', 0.30, 0.60, 0.85, ?
        WHERE NOT EXISTS (SELECT 1 FROM threshold_config WHERE id = 1)
        """,
        (time.time(),),
    )
    _ensure_alert_columns(conn)
    conn.commit()


def _ensure_alert_columns(conn: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(alerts)").fetchall()
    }
    if "acknowledged_by" not in columns:
        conn.execute("ALTER TABLE alerts ADD COLUMN acknowledged_by TEXT")
    if "ack_note" not in columns:
        conn.execute("ALTER TABLE alerts ADD COLUMN ack_note TEXT")
    if "acknowledged_at" not in columns:
        conn.execute("ALTER TABLE alerts ADD COLUMN acknowledged_at REAL")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


def _row_to_alert_record(row: sqlite3.Row) -> AlertRecord:
    return AlertRecord(
        id=int(row["id"]),
        timestamp=float(row["timestamp"]),
        camera_id=str(row["camera_id"]),
        risk_level=str(row["risk_level"]),
        score=float(row["score"]),
        evidence_window=(int(row["evidence_start"]), int(row["evidence_end"])),
        acknowledged_by=row["acknowledged_by"],
        ack_note=row["ack_note"],
        acknowledged_at=row["acknowledged_at"],
    )


def create_app(db_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Crowd Alert Service", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from src.api.demo_router import router as demo_router
    app.include_router(demo_router)

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
        row = app.state.db.execute(
            """
            SELECT id, timestamp, camera_id, risk_level, score, evidence_start, evidence_end,
                   acknowledged_by, ack_note, acknowledged_at
            FROM alerts
            WHERE id = ?
            """,
            (alert_id,),
        ).fetchone()
        return _row_to_alert_record(row)

    @app.get("/alerts")
    def list_alerts(limit: int = 50) -> AlertListResponse:
        rows = app.state.db.execute(
            """
            SELECT id, timestamp, camera_id, risk_level, score, evidence_start, evidence_end,
                   acknowledged_by, ack_note, acknowledged_at
            FROM alerts
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        alerts = [_row_to_alert_record(r) for r in rows]
        return AlertListResponse(alerts=alerts)

    @app.post("/alerts/{alert_id}/ack")
    def acknowledge_alert(alert_id: int, payload: AlertAckRequest) -> AlertRecord:
        cursor = app.state.db.execute(
            """
            UPDATE alerts
            SET acknowledged_by = ?, ack_note = ?, acknowledged_at = ?
            WHERE id = ?
            """,
            (payload.operator_name, payload.note, time.time(), alert_id),
        )
        app.state.db.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Alert not found")
        row = app.state.db.execute(
            """
            SELECT id, timestamp, camera_id, risk_level, score, evidence_start, evidence_end,
                   acknowledged_by, ack_note, acknowledged_at
            FROM alerts
            WHERE id = ?
            """,
            (alert_id,),
        ).fetchone()
        return _row_to_alert_record(row)

    @app.get("/config/thresholds")
    def get_threshold_config() -> ThresholdProfile:
        row = app.state.db.execute(
            "SELECT profile_name, low, medium, high FROM threshold_config WHERE id = 1"
        ).fetchone()
        return ThresholdProfile(
            profile_name=row["profile_name"],
            low=float(row["low"]),
            medium=float(row["medium"]),
            high=float(row["high"]),
        )

    @app.put("/config/thresholds")
    def update_threshold_config(payload: ThresholdProfile) -> ThresholdProfile:
        ordered = sorted([payload.low, payload.medium, payload.high])
        normalized = ThresholdProfile(
            profile_name=payload.profile_name,
            low=ordered[0],
            medium=ordered[1],
            high=ordered[2],
        )
        app.state.db.execute(
            """
            UPDATE threshold_config
            SET profile_name = ?, low = ?, medium = ?, high = ?, updated_at = ?
            WHERE id = 1
            """,
            (
                normalized.profile_name,
                normalized.low,
                normalized.medium,
                normalized.high,
                time.time(),
            ),
        )
        app.state.db.commit()
        return normalized

    @app.get("/dashboard/summary")
    def dashboard_summary(limit: int = 100, camera_id: str | None = None) -> DashboardSummaryResponse:
        params: list[Any] = []
        where_clause = ""
        if camera_id:
            where_clause = "WHERE camera_id = ?"
            params.append(camera_id)

        rows = app.state.db.execute(
            f"""
            SELECT id, timestamp, camera_id, risk_level, score, evidence_start, evidence_end,
                   acknowledged_by, ack_note, acknowledged_at
            FROM alerts
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        alerts = [_row_to_alert_record(r) for r in rows]

        timeline = [
            TimelinePoint(
                timestamp=float(r["timestamp"]),
                score=float(r["score"]),
                risk_level=str(r["risk_level"]),
                camera_id=str(r["camera_id"]),
            )
            for r in rows
        ]

        count_row = app.state.db.execute(
            f"""
            SELECT
                SUM(CASE WHEN risk_level = 'LOW' THEN 1 ELSE 0 END) AS low_count,
                SUM(CASE WHEN risk_level = 'MEDIUM' THEN 1 ELSE 0 END) AS medium_count,
                SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) AS high_count
            FROM alerts
            {where_clause}
            """,
            tuple(params),
        ).fetchone()

        level_counts = {
            "low": int(count_row["low_count"] or 0),
            "medium": int(count_row["medium_count"] or 0),
            "high": int(count_row["high_count"] or 0),
        }

        return DashboardSummaryResponse(
            alerts=alerts,
            timeline=timeline,
            level_counts=level_counts,
        )

    return app


app = create_app()
