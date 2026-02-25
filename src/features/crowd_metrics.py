from __future__ import annotations

import numpy as np


def _flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return np.linalg.norm(flow, axis=-1)


def _directional_entropy(flow: np.ndarray, bins: int = 16) -> float:
    if flow.size == 0:
        return 0.0
    angles = np.arctan2(flow[..., 1], flow[..., 0])
    angles = np.mod(angles, 2.0 * np.pi)
    hist, _ = np.histogram(angles, bins=bins, range=(0.0, 2.0 * np.pi), density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / float(total)
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(bins))


def _divergence_proxy(flow: np.ndarray) -> float:
    if flow.size == 0:
        return 0.0
    u = flow[..., 0]
    v = flow[..., 1]
    du_dx = np.gradient(u, axis=2)
    dv_dy = np.gradient(v, axis=1)
    divergence = du_dx + dv_dy
    return float(np.mean(np.abs(divergence)))


def compute_motion_statistics(flow: np.ndarray) -> dict[str, float]:
    mags = _flow_magnitude(flow)
    return {
        "flow_mean_magnitude": float(np.mean(mags)) if mags.size else 0.0,
        "flow_variance_magnitude": float(np.var(mags)) if mags.size else 0.0,
        "flow_directional_entropy": _directional_entropy(flow),
        "flow_divergence_proxy": _divergence_proxy(flow),
    }


def _occupancy_mask(frames: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (T, H, W, 3)")
    if frames.dtype == np.uint8:
        gray = np.mean(frames.astype(np.float32) / 255.0, axis=-1)
    else:
        gray = np.mean(np.clip(frames, 0.0, 1.0), axis=-1)
    return gray > threshold


def compute_density_pressure(frames: np.ndarray, threshold: float = 0.2) -> float:
    mask = _occupancy_mask(frames, threshold=threshold)
    return float(np.mean(mask))


def compute_local_concentration_index(
    frames: np.ndarray, grid_size: int = 4, threshold: float = 0.2
) -> float:
    if grid_size <= 0:
        raise ValueError("grid_size must be > 0")
    mask = _occupancy_mask(frames, threshold=threshold).astype(np.float32)
    h, w = mask.shape[1], mask.shape[2]
    row_splits = np.array_split(np.arange(h), grid_size)
    col_splits = np.array_split(np.arange(w), grid_size)
    cvs: list[float] = []
    for t in range(mask.shape[0]):
        block_means = []
        for rows in row_splits:
            for cols in col_splits:
                block = mask[t][np.ix_(rows, cols)]
                block_means.append(float(block.mean()))
        mean_val = float(np.mean(block_means))
        std_val = float(np.std(block_means))
        cvs.append(std_val / (mean_val + 1e-6))
    return float(np.mean(cvs)) if cvs else 0.0


def build_feature_row(frames: np.ndarray, flow: np.ndarray) -> dict[str, float]:
    motion = compute_motion_statistics(flow)
    return {
        **motion,
        "density_pressure": compute_density_pressure(frames),
        "local_concentration_index": compute_local_concentration_index(frames),
    }
