# COCO-FairTrain

COCO-FairTrain is a research-oriented training framework built on top of the Ultralytics codebase. Its purpose is narrow and explicit: make object detection training select checkpoints by COCO AP during training, so research comparisons are less distorted by fixed-epoch evaluation.

The central change is the low-level checkpoint selection logic. Instead of training every model for the same number of epochs and evaluating only the final checkpoint afterwards, this framework can run COCO API evaluation during training and use `metrics/mAP50-95(B)` to decide when `best.pt` should be updated.

## Why This Matters for Research

Many object detection papers compare models under a fixed training schedule. A common workflow is simple: train every model for the same number of epochs, then call the COCO API on the final checkpoint. This looks fair on the surface because every model receives the same budget, but it can hide an important source of bias.

Different models do not converge at the same speed. A lightweight or easier-to-optimize model may reach its best COCO AP earlier, then gradually overfit or degrade before the fixed final epoch. Another model may improve more slowly and happen to peak near the final evaluation point. If both models are reported only at the final epoch, the first model is judged after it has already passed its best state, while the second model is judged closer to its peak. The comparison then mixes model quality with convergence timing.

This is especially harmful in ablation studies and architecture comparisons. A method can look weaker not because its peak detection ability is lower, but because its best checkpoint occurred earlier than the chosen final epoch. The reported COCO AP becomes partly a result of schedule alignment, patience settings, and overfitting timing. For research claims, that is a fragile protocol.

COCO-FairTrain addresses this by aligning checkpoint selection with the same metric used for reporting. When enabled, the training loop periodically exports predictions, evaluates them with `pycocotools.COCOeval`, writes COCO `mAP50-95(B)` back into the trainer fitness, and only lets valid COCO-evaluated epochs update `best.pt`. The result is a training framework that is better suited to fair model-wise peak comparison.

## What This Fork Changes

- Runs official COCO API evaluation during selected validation epochs.
- Uses COCO `mAP50-95(B)` as training fitness when COCO evaluation succeeds.
- Prevents ordinary non-COCO validation epochs from overwriting a COCO-selected `best.pt`.
- Supports delayed COCO evaluation, so early training can skip expensive API calls.
- Supports common custom COCO annotation layouts.
- Maps prediction `image_id` values from the annotation JSON to avoid invalid COCO evaluation on custom datasets.
- Falls back to standard `torch.optim.SGD` instead of requiring a custom optimizer path.

For code-level implementation details, see the [implementation notes](ultralytics/%E6%9B%B4%E6%94%B9%E8%AF%B4%E6%98%8E.md).

## Modified Core Files

| Area | File | Role |
| --- | --- | --- |
| Checkpoint control | [`ultralytics/engine/trainer.py`](ultralytics/engine/trainer.py) | Reads COCO fitness, guards `best.pt`, and preserves COCO-related resume options. |
| Validation scheduling | [`ultralytics/engine/validator.py`](ultralytics/engine/validator.py) | Decides which epochs export predictions and run COCO API evaluation. |
| COCO metrics | [`ultralytics/models/yolo/detect/val.py`](ultralytics/models/yolo/detect/val.py) | Locates annotation JSON files, maps image IDs, calls `COCOeval`, and writes AP metrics. |
| Configuration | [`ultralytics/cfg/default.yaml`](ultralytics/cfg/default.yaml) | Adds switches for COCO-aligned training behavior. |
| Training entry | [`ultralytics/train.py`](ultralytics/train.py) | Example local entry point for the modified workflow. |

## New Options

```yaml
use_coco_fitness: False
coco_eval_interval: 1
coco_only_best: False
coco_start_epoch: 0
```

| Option | Meaning |
| --- | --- |
| `use_coco_fitness` | Enable training-time COCO API evaluation and use COCO AP as fitness when available. |
| `coco_eval_interval` | Run COCO API evaluation every N epochs. The final epoch is also evaluated. |
| `coco_only_best` | Only allow epochs with successful COCO evaluation to update `best.pt`. |
| `coco_start_epoch` | Start COCO API evaluation after a chosen epoch to reduce early overhead. |
| `save_json` | Required for COCO evaluation because predictions must be exported. It is enabled automatically when `use_coco_fitness=True`. |

## Recommended Research Protocol

Use the same training budget across models, but select each model's checkpoint by COCO AP observed during training:

1. Enable `use_coco_fitness=True`.
2. Set `coco_eval_interval` according to available compute, for example every 5 epochs.
3. Set `coco_only_best=True` so non-COCO epochs cannot replace the COCO-selected best checkpoint.
4. Use the resulting `best.pt` for final reporting or downstream testing.

This protocol keeps the fixed training budget while reducing the unfairness introduced by final-epoch-only evaluation. It is particularly useful when comparing architectures, loss designs, training tricks, or ablation variants whose convergence speed may differ.

## Installation

```bash
pip install -e .
pip install pycocotools
```

## Example

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

In this setup, checkpoints are judged by official COCO `mAP50-95(B)` whenever scheduled COCO evaluation runs. Intermediate validation epochs still provide normal training feedback, but they do not replace the COCO-selected `best.pt` when `coco_only_best=True`.

## Custom COCO Annotation Support

The validator searches common annotation locations:

```text
{data_path}/instances_val2017.json
{data_path}/annotations/instances_val2017.json
{data_path}/annotations/instances_val.json
{data_path}/annotations/instances_{split}.json
{data_path}/val/_annotations.coco.json
{data_path}/instances_val.json
{data_path}/_annotations.coco.json
```

It also builds an image ID map from the annotation file:

```python
self.img_id_map[Path(img["file_name"]).name] = img["id"]
self.img_id_map[Path(img["file_name"]).stem] = img["id"]
```

This matters because COCO prediction JSON must use the same `image_id` values as the ground-truth annotation file. Filename conversion alone is often unreliable for real custom datasets.

## Notes

- This is a personal modified fork of Ultralytics, not an official Ultralytics release.
- Upstream copyright notices and the GNU AGPL-3.0 license are retained.
- If `pycocotools` is unavailable, COCO API evaluation is skipped with a warning.
- The repository is intentionally focused on COCO-aligned checkpoint selection for research benchmarking.
