# CDP-Train

Technical fork of [Ultralytics](https://github.com/ultralytics/ultralytics) for COCO-AP-based best checkpoint selection during training.

This repository is not the independent CDP research-code repository. It is a modified Ultralytics codebase whose main purpose is to change the low-level training and validation logic that decides when `best.pt` should be updated.

## What This Fork Changes

Upstream Ultralytics training can validate at each epoch and save `best.pt` according to the training fitness value. This fork adds an optional COCO API path so that `best.pt` can be selected by the same COCO metric commonly used for object-detection reporting: `metrics/mAP50-95(B)`.

In practice, this means:

- Training can run official `pycocotools.COCOeval` during selected validation epochs.
- COCO `mAP50-95(B)` can be written back into the trainer's `fitness`.
- When COCO evaluation is scheduled every N epochs, non-COCO epochs can be blocked from replacing `best.pt`.
- Custom COCO-style annotation JSON files are searched in several common dataset layouts.
- Prediction `image_id` values can be mapped from the annotation JSON instead of relying only on filename-to-integer conversion.
- The optimizer fallback path avoids a hard dependency on `MuSGD` by using standard `torch.optim.SGD`.

For the detailed implementation notes, see [`ultralytics/更改说明.md`](ultralytics/%E6%9B%B4%E6%94%B9%E8%AF%B4%E6%98%8E.md).

## Modified Areas

| Area | File | Purpose |
| --- | --- | --- |
| Trainer checkpoint logic | [`ultralytics/engine/trainer.py`](ultralytics/engine/trainer.py) | Uses COCO fitness for `best.pt` and prevents non-COCO epochs from overwriting the best checkpoint when requested. |
| Validation scheduling | [`ultralytics/engine/validator.py`](ultralytics/engine/validator.py) | Enables JSON export only on scheduled COCO evaluation epochs and writes COCO fitness back to validation stats. |
| COCO evaluation | [`ultralytics/models/yolo/detect/val.py`](ultralytics/models/yolo/detect/val.py) | Calls `pycocotools`, locates annotation JSON files, maps image IDs, and records COCO AP metrics. |
| Default options | [`ultralytics/cfg/default.yaml`](ultralytics/cfg/default.yaml) | Adds switches for COCO fitness and scheduled COCO evaluation. |
| Example entry point | [`ultralytics/train.py`](ultralytics/train.py) | Provides a local training example using the modified training flow. |
| Utility scripts | [`coco-test.py`](coco-test.py), [`visual.py`](visual.py) | Helper scripts for COCO testing and visualization. |

## New Training Options

The fork adds these options to `ultralytics/cfg/default.yaml`:

```yaml
use_coco_fitness: False
coco_eval_interval: 1
coco_only_best: False
coco_start_epoch: 0
```

| Option | Meaning |
| --- | --- |
| `use_coco_fitness` | Enable COCO API evaluation during training validation and use COCO AP as fitness when available. |
| `coco_eval_interval` | Run COCO API evaluation every N epochs. The final epoch is also evaluated. |
| `coco_only_best` | If enabled, only epochs that actually ran COCO evaluation may update `best.pt`. |
| `coco_start_epoch` | Skip COCO API evaluation before this epoch to reduce early training overhead. |
| `save_json` | Required for COCO API evaluation because predictions must be exported to JSON. The trainer enables it automatically when `use_coco_fitness=True`. |

## Recommended Use

Install the repository in editable mode and install COCO API support:

```bash
pip install -e .
pip install pycocotools
```

Example training call:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8s.yaml")

results = model.train(
    data="path/to/data.yaml",
    epochs=250,
    imgsz=640,
    seed=0,
    deterministic=True,
    save_json=True,
    use_coco_fitness=True,
    coco_eval_interval=5,
    coco_only_best=True,
    coco_start_epoch=100,
    patience=100,
)
```

With this setup, `best.pt` is updated only from epochs where COCO API evaluation has produced a valid `metrics/mAP50-95(B)` value. This is useful when the checkpoint used for later testing or comparison should be aligned with official COCO AP instead of an intermediate training signal.

## COCO Annotation Compatibility

For custom COCO-style datasets, the detection validator searches common annotation locations, including:

```text
{data_path}/instances_val2017.json
{data_path}/annotations/instances_val2017.json
{data_path}/annotations/instances_val.json
{data_path}/annotations/instances_{split}.json
{data_path}/val/_annotations.coco.json
{data_path}/instances_val.json
{data_path}/_annotations.coco.json
```

It also builds an `image_id` lookup from the annotation file:

```python
self.img_id_map[Path(img["file_name"]).name] = img["id"]
self.img_id_map[Path(img["file_name"]).stem] = img["id"]
```

This avoids incorrect COCO evaluation when image filenames cannot be safely converted into integer IDs.

## Checkpoint Selection Behavior

When `use_coco_fitness=True` and `coco_only_best=True`, the trainer handles validation epochs as follows:

| Epoch type | COCO API run? | Can update `best.pt`? |
| --- | --- | --- |
| Scheduled COCO epoch | Yes | Yes, if COCO `mAP50-95(B)` improves. |
| Final epoch | Yes | Yes, if COCO `mAP50-95(B)` improves. |
| Ordinary validation epoch between scheduled COCO evaluations | No | No. |

This prevents ordinary validation metrics from replacing a checkpoint selected by official COCO AP.

## Notes

- This is a personal modified fork, not an official Ultralytics release.
- Upstream Ultralytics copyright notices and the GNU AGPL-3.0 license are retained.
- If `pycocotools` is not installed, COCO API evaluation is skipped with a warning.
- For exact code-level changes, start with [`ultralytics/更改说明.md`](ultralytics/%E6%9B%B4%E6%94%B9%E8%AF%B4%E6%98%8E.md).
