import numpy as np

from src.data.clips import generate_clip_windows


def test_generate_clip_windows_count_and_shape() -> None:
    frames = np.random.default_rng(4).random((10, 24, 32, 3), dtype=np.float32)

    clips = generate_clip_windows(frames, clip_length_frames=4, clip_stride_frames=2)

    assert clips.shape == (4, 4, 24, 32, 3)


def test_generate_clip_windows_handles_short_inputs() -> None:
    frames = np.random.default_rng(9).random((3, 24, 32, 3), dtype=np.float32)

    clips = generate_clip_windows(frames, clip_length_frames=5, clip_stride_frames=1)

    assert clips.shape == (0, 5, 24, 32, 3)
