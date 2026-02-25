from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.client import DashboardApiClient

st.set_page_config(page_title="Crowd Alert Dashboard", layout="wide")


def _profile_defaults(profile_name: str) -> tuple[float, float, float]:
    if profile_name == "strict":
        return 0.35, 0.65, 0.90
    if profile_name == "relaxed":
        return 0.25, 0.50, 0.80
    return 0.30, 0.60, 0.85


def main() -> None:
    st.title("Abnormal Crowd Behaviour Monitoring")
    st.caption("Live risk monitoring, threshold controls, and alert acknowledgment workflow")

    with st.sidebar:
        st.header("Connection")
        api_base_url = st.text_input("API Base URL", value="http://127.0.0.1:8000")
        limit = int(st.slider("Alert Window", min_value=20, max_value=500, value=100, step=10))
        profile_name = st.selectbox("Threshold Profile", ["default", "strict", "relaxed", "custom"])

    client = DashboardApiClient(api_base_url)
    try:
        health = client.health()
        if health.get("status") != "ok":
            st.error("API health check failed.")
            return
    except Exception as exc:  # pragma: no cover - runtime UI guard
        st.error(f"Unable to connect to API: {exc}")
        return

    summary = client.summary(limit=limit)
    alerts = summary.get("alerts", [])
    timeline = summary.get("timeline", [])
    level_counts = summary.get("level_counts", {"low": 0, "medium": 0, "high": 0})

    if alerts:
        camera_options = sorted({a["camera_id"] for a in alerts})
        selected_camera = st.selectbox("Camera", ["ALL", *camera_options], index=0)
    else:
        selected_camera = "ALL"

    if selected_camera != "ALL":
        summary = client.summary(limit=limit, camera_id=selected_camera)
        alerts = summary.get("alerts", [])
        timeline = summary.get("timeline", [])
        level_counts = summary.get("level_counts", {"low": 0, "medium": 0, "high": 0})

    col1, col2, col3 = st.columns(3)
    col1.metric("LOW Alerts", level_counts.get("low", 0))
    col2.metric("MEDIUM Alerts", level_counts.get("medium", 0))
    col3.metric("HIGH Alerts", level_counts.get("high", 0))

    st.subheader("Threshold Controls")
    if profile_name == "custom":
        current = client.get_thresholds()
        default_low = float(current["low"])
        default_medium = float(current["medium"])
        default_high = float(current["high"])
    else:
        default_low, default_medium, default_high = _profile_defaults(profile_name)

    tcol1, tcol2, tcol3 = st.columns(3)
    low = tcol1.slider("Low Threshold", 0.0, 1.0, value=float(default_low), step=0.01)
    medium = tcol2.slider("Medium Threshold", 0.0, 1.0, value=float(default_medium), step=0.01)
    high = tcol3.slider("High Threshold", 0.0, 1.0, value=float(default_high), step=0.01)
    if st.button("Apply Threshold Profile"):
        updated = client.update_thresholds(profile_name=profile_name, low=low, medium=medium, high=high)
        st.success(
            f"Thresholds applied: {updated['profile_name']} "
            f"({updated['low']:.2f}, {updated['medium']:.2f}, {updated['high']:.2f})"
        )

    st.subheader("Risk Timeline")
    if timeline:
        timeline_df = pd.DataFrame(timeline)
        timeline_df = timeline_df.sort_values("timestamp")
        st.line_chart(timeline_df.set_index("timestamp")[["score"]], height=240)
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)
    else:
        st.info("No timeline records available.")

    st.subheader("Alert History")
    if alerts:
        alerts_df = pd.DataFrame(alerts)
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts yet.")

    st.subheader("Operator Acknowledgment")
    with st.form("ack_form"):
        alert_id = st.number_input("Alert ID", min_value=1, step=1)
        operator_name = st.text_input("Operator Name", value="operator-1")
        note = st.text_input("Acknowledgment Note", value="Checked and escalated.")
        submit = st.form_submit_button("Acknowledge Alert")
        if submit:
            try:
                client.acknowledge_alert(int(alert_id), operator_name=operator_name, note=note)
                st.success(f"Alert {int(alert_id)} acknowledged.")
            except Exception as exc:  # pragma: no cover - runtime UI guard
                st.error(f"Failed to acknowledge alert: {exc}")


if __name__ == "__main__":
    main()
