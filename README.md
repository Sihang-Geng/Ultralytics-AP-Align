# Ultralytics YOLO COCO Fitness Fork

> This repository is a personal modified fork of [Ultralytics](https://github.com/ultralytics/ultralytics). It is not an official Ultralytics repository. The original Ultralytics copyright notices and the GNU AGPL-3.0 license are retained.

本仓库基于 Ultralytics YOLO 官方代码进行训练流程和评估流程改造，核心目标是在训练阶段直接接入 `pycocotools` COCO API，并使用 COCO `mAP50-95(B)` 参与 `best.pt` 的选择。它适合需要用 COCO 标注 JSON、COCO 官方指标和自定义数据集路径进行训练验证的场景。

## Core Changes

- Training-time COCO evaluation: 在训练验证阶段按 epoch 生成 `predictions.json`，并调用 `pycocotools` 计算 COCO mAP。
- COCO fitness selection: 支持将 COCO `metrics/mAP50-95(B)` 作为训练 `fitness`，用于控制 `best.pt` 保存。
- Evaluation schedule controls: 新增 `use_coco_fitness`、`coco_eval_interval`、`coco_only_best`、`coco_start_epoch` 等训练参数。
- Custom COCO annotation lookup: 支持多种常见 annotation JSON 路径，包括 `annotations/instances_val2017.json`、`instances_val.json`、`val/_annotations.coco.json`、`_annotations.coco.json` 等。
- Image ID mapping: 从 annotation JSON 读取 `images[].file_name` 和 `images[].id`，避免自定义图片名无法转换为 COCO `image_id`。
- Optimizer fallback: 移除训练流程对 `MuSGD` 的强依赖，自动优化器中的相关路径回退为标准 `torch.optim.SGD`。
- Example scripts and visualization: 保留训练、COCO 测试、可视化和绘图脚本，并提供效果图示例。

## Example Outputs

`visual.py`、`ultralytics/plotfig2.py` 等脚本的输出示例：

![Example 1](ultralytics/example1.jpg)

![Example 2](ultralytics/example2.jpg)

## Training Usage

示例训练入口位于 [ultralytics/train.py](ultralytics/train.py)。当前示例保留了本地训练路径，迁移到其他环境时可按自己的数据集路径修改。

```python
from ultralytics import YOLO

model = YOLO("/root/ultralytics/ultralytics/cfg/models/v8/yolov8s.yaml")

results = model.train(
    data="/root/ultralytics/ultralytics/cfg/RUOD/RUOD_YOLO/data.yaml",
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

推荐参数含义：

| Parameter | Description |
| --- | --- |
| `save_json=True` | 收集验证预测结果并生成 COCO API 所需的 JSON。 |
| `use_coco_fitness=True` | 启用训练阶段 COCO API 评估。 |
| `coco_eval_interval=10` | 每 10 个 epoch 执行一次 COCO API，降低训练开销。 |
| `coco_only_best=True` | 只允许实际完成 COCO API 的 epoch 更新 `best.pt`。 |
| `coco_start_epoch=100` | 前 100 个 epoch 跳过 COCO API，训练后期再使用 COCO 指标选择模型。 |

## Implementation Notes

训练阶段的主要流程如下：

1. `BaseTrainer` 读取训练参数，并在启用 `use_coco_fitness` 时确保 `save_json` 与 COCO 评估参数可用。
2. `BaseValidator` 根据 `coco_start_epoch` 和 `coco_eval_interval` 判断当前 epoch 是否需要运行 COCO API。
3. 需要评估时，验证器收集预测框并写出 `predictions.json`。
4. `DetectionValidator` 查找自定义 COCO annotation JSON，并根据 `images[].file_name` 建立 image id 映射。
5. `pycocotools.COCOeval` 计算 `mAP50`、`mAP50-95`、small/medium/large AP。
6. `metrics/mAP50-95(B)` 被写回训练 stats，并作为 COCO fitness 参与 `best.pt` 选择。
7. 当 `coco_only_best=True` 时，未运行 COCO API 的 epoch 不会覆盖基于 COCO 指标得到的最佳模型。

更多细节见 [ultralytics/更改说明.md](ultralytics/%E6%9B%B4%E6%94%B9%E8%AF%B4%E6%98%8E.md)。

## Important Files

| File | Purpose |
| --- | --- |
| [ultralytics/engine/trainer.py](ultralytics/engine/trainer.py) | COCO fitness 参数补全、resume 参数处理、best.pt 更新策略、SGD 回退。 |
| [ultralytics/engine/validator.py](ultralytics/engine/validator.py) | 训练验证阶段按 epoch 控制 `save_json`、写出 predictions、调用 COCO API。 |
| [ultralytics/models/yolo/detect/val.py](ultralytics/models/yolo/detect/val.py) | annotation JSON 查找、image_id 映射、`pycocotools` COCOeval 指标回写。 |
| [ultralytics/cfg/default.yaml](ultralytics/cfg/default.yaml) | 新增 COCO fitness 相关配置项。 |
| [ultralytics/train.py](ultralytics/train.py) | 当前训练示例。 |
| [coco-test.py](coco-test.py) | COCO 相关测试脚本。 |
| [visual.py](visual.py) | 可视化脚本。 |
| [ultralytics/plotfig2.py](ultralytics/plotfig2.py) | 绘图脚本。 |
| [ultralytics/3d.py](ultralytics/3d.py) | 3D/可视化相关脚本。 |

## Installation

基础安装方式与 Ultralytics 官方项目保持一致：

```bash
pip install -e .
```

如果需要训练阶段 COCO API 指标，请确保安装：

```bash
pip install pycocotools
```

## License

This project is released under the GNU AGPL-3.0 license inherited from Ultralytics. See [LICENSE](LICENSE).

Ultralytics upstream repository: <https://github.com/ultralytics/ultralytics>
