from __future__ import annotations

from typing import Any

import httpx


class DashboardApiClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(method=method, url=url, **kwargs)
            response.raise_for_status()
            return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def summary(self, limit: int = 100, camera_id: str | None = None) -> dict[str, Any]:
        params = {"limit": limit}
        if camera_id:
            params["camera_id"] = camera_id
        return self._request("GET", "/dashboard/summary", params=params)

    def get_thresholds(self) -> dict[str, Any]:
        return self._request("GET", "/config/thresholds")

    def update_thresholds(
        self, profile_name: str, low: float, medium: float, high: float
    ) -> dict[str, Any]:
        payload = {
            "profile_name": profile_name,
            "low": low,
            "medium": medium,
            "high": high,
        }
        return self._request("PUT", "/config/thresholds", json=payload)

    def acknowledge_alert(self, alert_id: int, operator_name: str, note: str) -> dict[str, Any]:
        payload = {"operator_name": operator_name, "note": note}
        return self._request("POST", f"/alerts/{alert_id}/ack", json=payload)
