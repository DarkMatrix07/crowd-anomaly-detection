from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class NormalClipDataset(Dataset):
    """Loads overlapping fixed-length clips from directories of JPEG frames.

    Each directory represents one video (frames named numerically, e.g. 0001.jpg).
    Used for training the CNN autoencoder on normal-only footage.

    Output clips have pixel values normalized to [0, 1].

    Parameters
    ----------
    video_dirs : list of directories, each containing numbered .jpg frames
    clip_len   : number of frames per clip (T)
    frame_size : (width, height) to resize each frame to
    stride     : step between clip start positions (overlap = clip_len - stride)
    """

    def __init__(
        self,
        video_dirs: list[Path],
        clip_len: int,
        frame_size: tuple[int, int],
        stride: int,
    ) -> None:
        self._clip_len = clip_len
        self._frame_size = frame_size
        self._clips: list[tuple[list[Path], int]] = []  # (frame_paths, start_idx)

        for d in video_dirs:
            frames = sorted(Path(d).glob("*.jpg"), key=lambda p: int(p.stem))
            if len(frames) < clip_len:
                continue
            for start in range(0, len(frames) - clip_len + 1, stride):
                self._clips.append((frames, start))

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> np.ndarray:
        frames_list, start = self._clips[idx]
        clip = []
        w, h = self._frame_size
        for frame_path in frames_list[start : start + self._clip_len]:
            img = cv2.imread(str(frame_path))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            clip.append(img.astype(np.float32) / 255.0)
        arr = np.stack(clip, axis=0)          # (T, H, W, 3)
        return arr.transpose(3, 0, 1, 2)      # (3, T, H, W)
