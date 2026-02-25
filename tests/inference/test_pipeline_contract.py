import numpy as np

from src.inference.pipeline import RollingInferencePipeline


def test_pipeline_emits_alert_event_contract() -> None:
    rng = np.random.default_rng(12)
    frames = [rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8) for _ in range(10)]

    def dummy_model_score(_: np.ndarray) -> float:
        return 0.9

    pipeline = RollingInferencePipeline(
        camera_id="cam-1",
        clip_length=4,
        clip_stride=2,
        anomaly_model_fn=dummy_model_score,
    )

    events = []
    for idx, frame in enumerate(frames):
        events.extend(pipeline.process_frame(frame=frame, timestamp=float(idx) * 0.1))

    assert len(events) > 0
    required_keys = {"timestamp", "camera_id", "risk_level", "score", "evidence_window"}
    assert required_keys.issubset(events[0].keys())
