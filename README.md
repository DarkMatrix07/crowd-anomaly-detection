# Abnormal Crowd Behaviour Detection

Implementation scaffold for abnormal crowd behaviour detection and early warning.

## Prerequisites

- Python 3.11+

## Setup

```bash
python -m pip install -r requirements.txt
```

## CLI Usage

```bash
python -m src --help
python -m src test
python -m src train --config configs/train.yaml
python -m src infer --source path/to/video.mp4 --config configs/infer.yaml
```

## Notes

- `train` and `infer` are scaffolds in this phase and will be wired in later tasks.
