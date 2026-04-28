<div align="center">

# CDP-train

### COCO-driven peak checkpoint selection for fairer Ultralytics YOLO research comparisons.

<p>
  <a href="https://github.com/Sihang-Geng/CDP-train/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-AGPL--3.0-blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.8%2B-3776AB?logo=python&logoColor=white">
  <img alt="YOLO" src="https://img.shields.io/badge/base-Ultralytics%20YOLO-111111">
  <img alt="Metric" src="https://img.shields.io/badge/metric-COCO%20AP%20aligned-2E7D32">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20code-orange">
</p>

<p>
  <b>COCO-driven peak checkpoint selection</b> |
  <b>fair model comparison</b> |
  <b>AP-aligned training evaluation</b> |
  <b>paper-style visualization scripts</b>
</p>

</div>

> **Notice**
> This repository is a personal modified fork of [Ultralytics](https://github.com/ultralytics/ultralytics), not an official Ultralytics repository. The upstream copyright notices and the GNU AGPL-3.0 license are retained.

## Motivation

In object detection research, model comparison is often performed after training all models for a fixed number of epochs and then running an evaluation API on the final checkpoint. This protocol is simple, but it can be unfair when different models converge at different speeds.

An early-converging model may reach its best performance before the final epoch and then begin to overfit. If only the last checkpoint is evaluated, its reported AP may be lower than its actual peak capability. A slower-converging model may show a different behavior under the same fixed-length setting. As a result, comparing only the final checkpoint can mix model quality with convergence timing, which weakens the fairness of ablation studies and cross-model comparisons.

This issue is especially important in papers that report COCO-style AP metrics, such as `AP@[.50:.95]` / `mAP50-95`. If the paper reports COCO AP, then checkpoint selection during training should also be aligned with COCO AP rather than relying only on a fixed final epoch.

`CDP-train` is designed for this research setting. It modifies the Ultralytics training process so that COCO evaluation can be run periodically during training, and the checkpoint with the best COCO score is retained. In this way, each model is compared closer to its own peak performance, making the comparison less sensitive to convergence speed and late-stage overfitting.

In short, `CDP-train` focuses on the following question:

> If different models converge at different times, can we compare them by their best COCO AP during training rather than only by the final fixed-epoch checkpoint?

This repository opens the non-core training, evaluation, and visualization support code from an ongoing manuscript. The full research method and complete framework diagram are not released here because the paper has not been formally accepted yet. After acceptance, the core framework figure and additional method details will be supplemented.

## What This Repository Solves

| Research workflow problem | What CDP-train provides |
| --- | --- |
| Fixed-epoch evaluation can favor or penalize models depending on convergence speed. | Keeps the checkpoint with the best COCO score observed during training. |
| Early-converging models may overfit before the final epoch. | Periodically evaluates COCO AP and preserves the peak checkpoint. |
| Final papers report COCO AP, but training may select checkpoints with a different fitness signal. | Uses COCO `mAP50-95(B)` as the training fitness for `best.pt`. |
| Running full COCO API every epoch is expensive. | Adds scheduled evaluation with `coco_eval_interval` and `coco_start_epoch`. |
| Custom COCO-format datasets often do not follow official COCO folder naming. | Searches multiple common annotation JSON locations. |
| Custom image names may not be numeric COCO IDs. | Reads `images[].file_name` and `images[].id` from annotation JSON for reliable `image_id` mapping. |
| Reproducing paper figures requires more than training code. | Includes plotting and visualization scripts with example outputs. |
| Some environments do not include custom optimizer dependencies. | Falls back from `MuSGD` to standard `torch.optim.SGD`. |

## Figure 8: Plotting Script Output

The figure below is an example produced by the released plotting code. It is placed here intentionally: the repository includes not only training modifications, but also the figure-generation utilities used around the research workflow.

<p align="center">
  <img src="ultralytics/example2.jpg" alt="Figure 8 visualization produced by the plotting scripts" width="92%">
</p>

<p align="center">
  <sub><b>Fig. 8.</b> Example output from the released paper-style plotting utilities.</sub>
</p>

## Released Scope

This is a partial research-code release. The goal is to make the training and evaluation support code inspectable while keeping unpublished core method details outside the repository.

| Component | Status | Notes |
| --- | --- | --- |
| COCO-driven peak checkpoint selection | Released | Keeps the best COCO checkpoint observed during training. |
| Training-time COCO API evaluation | Released | Runs `pycocotools.COCOeval` during selected validation epochs. |
| Custom COCO annotation lookup | Released | Supports multiple common JSON layouts. |
| Annotation-based image ID mapping | Released | Aligns prediction `image_id` with ground-truth COCO JSON. |
| Visualization and plotting scripts | Released | Includes qualitative and paper-style figure utilities. |
| Complete unpublished method and framework diagram | Not released | Reserved until the manuscript is formally accepted. |

## Main Features

### 1. COCO-driven peak checkpoint selection

The central modification is to make checkpoint saving follow the best COCO score observed during training. When COCO evaluation is executed successfully, the validator writes `metrics/mAP50-95(B)` back into the training stats and uses it as `fitness`.

Instead of comparing models only at a fixed final training length, this allows each model to be evaluated near its own best validation point. This is useful for ablation studies because convergence speed, overfitting timing, and late-stage instability can otherwise distort the comparison.

### 2. Scheduled COCO evaluation during training

Full COCO evaluation requires prediction collection, JSON export, annotation loading, and COCOeval accumulation. To keep training efficient, CDP-train runs COCO evaluation periodically rather than at every epoch. In the paper setting, COCO evaluation was run every five epochs, and the checkpoint with the best COCO score was kept.

```yaml
use_coco_fitness: False
coco_eval_interval: 1
coco_only_best: False
coco_start_epoch: 0
```

| Parameter | Purpose |
| --- | --- |
| `use_coco_fitness` | Enables COCO-driven checkpoint selection during training. |
| `coco_eval_interval` | Runs COCO API every N epochs instead of every epoch. |
| `coco_only_best` | Prevents non-COCO epochs from updating `best.pt`. |
| `coco_start_epoch` | Skips early COCO evaluation to reduce warmup-stage overhead. |

### 3. Custom COCO JSON support

The modified detection validator searches several common annotation locations:

```text
{data_path}/instances_val2017.json
{data_path}/annotations/instances_val2017.json
{data_path}/annotations/instances_val.json
{data_path}/annotations/instances_{split}.json
{data_path}/val/_annotations.coco.json
{data_path}/instances_val.json
{data_path}/_annotations.coco.json
```

This is useful for datasets exported from labeling platforms or arranged in COCO-like, but not official COCO-identical, directory layouts.

### 4. Annotation-based image ID mapping

COCO prediction JSON must use the same `image_id` values as the ground-truth file. The validator therefore builds an ID map from the annotation JSON:

```python
self.img_id_map[Path(img["file_name"]).name] = img["id"]
self.img_id_map[Path(img["file_name"]).stem] = img["id"]
```

This avoids incorrect AP results when validation images use names such as `ship_0001.jpg`, exported filenames, or other non-numeric identifiers.

### 5. Visualization and plotting utilities

The repository also includes scripts used for qualitative inspection and figure preparation. This is useful because research code often requires both metric-aligned training and clear visual evidence for analysis.

## Quick Start

Install the project in editable mode:

```bash
pip install -e .
```

Install COCO evaluation support:

```bash
pip install pycocotools
```

Run the included training example after adapting paths for your environment:

```bash
python ultralytics/train.py
```

Minimal training example:

```python
from ultralytics import YOLO

model = YOLO("/root/ultralytics/ultralytics/cfg/models/v8/yolov8s.yaml")

results = model.train(
    data="/root/ultralytics/ultralytics/cfg/datasets/RUOD/RUOD_YOLO/data.yaml",
    epochs=250,
    imgsz=640,
    seed=0,
    deterministic=True,
    save_json=True,
    use_coco_fitness=True,
    coco_eval_interval=10,
    coco_only_best=True,
    coco_start_epoch=100,
    patience=100,
)

results = model.val()
```

## Recommended Training Modes

### CDP-style fair comparison training

Use this mode when the reported metric is COCO AP and model comparison should use the best COCO checkpoint observed during training:

```python
model.train(
    save_json=True,
    use_coco_fitness=True,
    coco_eval_interval=5,
    coco_only_best=True,
    coco_start_epoch=100,
)
```

### Fast pipeline check

Use this mode when only checking whether training runs:

```python
model.train(
    save_json=False,
    use_coco_fitness=False,
)
```

## Qualitative Visualization Example

The following example is generated by the released visualization utilities and is intended for qualitative inspection.

<p align="center">
  <img src="ultralytics/example1.jpg" alt="Qualitative visualization example" width="92%">
</p>

<p align="center">
  <sub><b>Qualitative example.</b> Output from the released visualization script.</sub>
</p>

## Repository Map

| File | Role |
| --- | --- |
| [`ultralytics/engine/trainer.py`](ultralytics/engine/trainer.py) | Adds COCO-driven fitness handling, resume parameter support, `best.pt` control, and SGD fallback. |
| [`ultralytics/engine/validator.py`](ultralytics/engine/validator.py) | Controls training-time JSON collection and scheduled COCO API calls. |
| [`ultralytics/models/yolo/detect/val.py`](ultralytics/models/yolo/detect/val.py) | Implements custom annotation lookup, image ID mapping, and `pycocotools.COCOeval` metric writing. |
| [`ultralytics/cfg/default.yaml`](ultralytics/cfg/default.yaml) | Defines the COCO fitness configuration fields. |
| [`ultralytics/train.py`](ultralytics/train.py) | Training entry example for this fork. |
| [`coco-test.py`](coco-test.py) | COCO-related test script. |
| [`visual.py`](visual.py) | Qualitative visualization script. |
| [`ultralytics/plotfig2.py`](ultralytics/plotfig2.py) | Paper-style plotting script. |
| [`ultralytics/3d.py`](ultralytics/3d.py) | 3D visualization helper script. |
| [Change notes](ultralytics/%E6%9B%B4%E6%94%B9%E8%AF%B4%E6%98%8E.md) | Detailed technical notes for the code changes. |

## Technical Notes

<details>
<summary><b>How fitness is selected</b></summary>

When COCO evaluation is enabled and runs successfully, `metrics/mAP50-95(B)` is written back to the validation stats. The trainer reads this value as `fitness`. If `coco_only_best=True`, epochs that skip COCO evaluation are assigned `-inf` fitness so they cannot replace the best COCO-selected checkpoint.

</details>

<details>
<summary><b>Why COCO evaluation is scheduled</b></summary>

COCO API evaluation is more expensive than normal validation because it requires prediction collection, JSON writing, annotation loading, and full COCOeval accumulation. `coco_eval_interval` and `coco_start_epoch` reduce this overhead while still allowing the training process to keep the best COCO checkpoint rather than only the final fixed-epoch checkpoint.

</details>

<details>
<summary><b>What happens without pycocotools</b></summary>

If `pycocotools` is not installed, the validator prints a warning and skips COCO API evaluation instead of crashing the training process. Install it with `pip install pycocotools` when COCO-driven checkpoint selection is required.

</details>

## License

This project is released under the GNU AGPL-3.0 license inherited from Ultralytics. See [LICENSE](LICENSE).

Upstream project: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
