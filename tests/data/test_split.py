from pathlib import Path

from src.data.split import collect_video_files, split_video_files


def _touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-video")


def test_collect_video_files_finds_avi_only(tmp_path: Path) -> None:
    _touch_video(tmp_path / "01_001.avi")
    _touch_video(tmp_path / "01_002.mp4")
    (tmp_path / "readme.txt").write_text("x", encoding="utf-8")

    files = collect_video_files(tmp_path)

    assert [p.name for p in files] == ["01_001.avi", "01_002.mp4"]


def test_split_video_files_respects_70_30_ratio(tmp_path: Path) -> None:
    videos = []
    for idx in range(10):
        path = tmp_path / f"v_{idx:03d}.avi"
        _touch_video(path)
        videos.append(path)

    assignments = split_video_files(videos, train_ratio=0.7, seed=42)

    train_count = sum(1 for _, split in assignments if split == "train")
    test_count = sum(1 for _, split in assignments if split == "test")
    assert train_count == 7
    assert test_count == 3


def test_split_video_files_is_deterministic_with_seed(tmp_path: Path) -> None:
    videos = []
    for idx in range(20):
        path = tmp_path / f"v_{idx:03d}.avi"
        _touch_video(path)
        videos.append(path)

    first = split_video_files(videos, train_ratio=0.7, seed=7)
    second = split_video_files(videos, train_ratio=0.7, seed=7)

    assert first == second
