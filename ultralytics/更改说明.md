# 自定义 Ultralytics YOLO 改动说明

本文档记录当前仓库相对 Ultralytics 官方代码的主要技术改动。当前版本重点围绕训练阶段 COCO API 评估、COCO fitness 选择、自定义 COCO 标注适配、优化器回退以及示例脚本开源整理展开。

## 1. 改动目标

原始训练流程中，训练阶段的验证指标可以用于日志、早停和 `best.pt` 选择，但对于使用 COCO 格式 annotation JSON 的自定义数据集，默认流程并不总是直接等价于 `pycocotools` 的 COCO 官方评估口径。

本仓库的目标是：

1. 在训练验证阶段按需生成 COCO predictions JSON。
2. 在训练过程中调用 `pycocotools` 计算 COCO mAP。
3. 使用 COCO `mAP50-95(B)` 控制训练过程中的 `fitness` 和 `best.pt`。
4. 支持自定义 COCO annotation JSON 的多路径查找。
5. 支持从 annotation JSON 中映射真实 `image_id`，避免文件名转换造成评估错误。
6. 移除对 `MuSGD` 的强依赖，使用标准 `torch.optim.SGD` 兜底。
7. 将训练、COCO 测试、可视化和绘图脚本一起开源，方便复现实验流程和效果图。

## 2. 新增训练配置项

文件：`ultralytics/cfg/default.yaml`

新增参数：

```yaml
use_coco_fitness: False
coco_eval_interval: 1
coco_only_best: False
coco_start_epoch: 0
```

参数含义：

| 参数 | 作用 |
| --- | --- |
| `use_coco_fitness` | 是否启用训练阶段 COCO API 评估，并使用 COCO 指标参与 fitness。 |
| `coco_eval_interval` | 每隔多少个 epoch 调用一次 COCO API。 |
| `coco_only_best` | 是否只允许完成 COCO API 的 epoch 更新 `best.pt`。 |
| `coco_start_epoch` | 从第几个 epoch 开始执行 COCO API，前期可跳过以减少开销。 |

## 3. BaseTrainer 改动

文件：`ultralytics/engine/trainer.py`

### 3.1 COCO 参数补全

在 `BaseTrainer.__init__` 中，如果启用 `use_coco_fitness`，会确保：

1. `save_json=True`
2. `coco_eval_interval` 有默认值
3. `coco_only_best` 有默认值

这样可以避免用户开启 COCO fitness 后忘记生成 predictions JSON。

### 3.2 best.pt 更新策略

`BaseTrainer.validate()` 从验证结果中读取内部标记：

```python
coco_eval = metrics.pop("coco_eval", 0.0)
```

当 `use_coco_fitness=True` 且 `coco_only_best=True` 时，如果当前 epoch 没有实际执行 COCO API，则将 fitness 设为 `-inf`，阻止这一轮覆盖 `best.pt`。

这样做的原因是：当 COCO API 每隔 N 个 epoch 才运行一次时，中间普通验证轮次不应该覆盖基于 COCO mAP 选出的最佳模型。

### 3.3 resume 参数

恢复训练时允许覆盖以下参数：

```python
epochs
save_json
use_coco_fitness
coco_eval_interval
coco_only_best
coco_start_epoch
```

这样在 resume 时可以调整 COCO 评估策略。

### 3.4 MuSGD 回退

训练器中去掉了对 `ultralytics.optim.MuSGD` 的强依赖。自动优化器中原本可能选择 `MuSGD` 的路径改为标准 `SGD`：

```python
name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
```

如果用户显式传入 `MuSGD`，也会转换为 `SGD`。这样可以避免环境中没有自定义优化器时训练流程直接失败。

## 4. BaseValidator 改动

文件：`ultralytics/engine/validator.py`

### 4.1 按 epoch 控制 save_json

训练验证开始时会根据当前 epoch 判断是否运行 COCO API：

```python
coco_eval_this_epoch = (
    ((trainer.epoch + 1) >= start_epoch)
    and (
        (eval_interval <= 1)
        or ((trainer.epoch + 1) % eval_interval == 0)
        or ((trainer.epoch + 1) == trainer.epochs)
    )
)
```

如果本轮不需要 COCO API，则临时关闭 `self.args.save_json`，减少每个 epoch 都生成 JSON 的开销。

### 4.2 写出 predictions.json

当当前 epoch 需要 COCO API 且 `self.jdict` 非空时，验证器会将预测结果保存到：

```text
{save_dir}/predictions.json
```

然后调用：

```python
stats = self.eval_json(stats)
```

这个函数由检测验证器负责实现。

### 4.3 COCO fitness 回写

如果 `eval_json()` 返回了：

```python
metrics/mAP50-95(B)
```

则训练验证阶段将其写入：

```python
stats["fitness"] = coco_fitness
stats["coco_eval"] = 1.0
```

`coco_eval` 是内部标记，用于通知训练器这一轮可以参与基于 COCO 指标的 `best.pt` 更新。

## 5. DetectionValidator 改动

文件：`ultralytics/models/yolo/detect/val.py`

### 5.1 pycocotools 接入

当前实现使用：

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
```

如果没有安装 `pycocotools`，训练不会直接中断，而是打印 warning 并跳过 COCO API 评估。

### 5.2 annotation JSON 路径搜索

除标准 COCO/LVIS 路径外，自定义数据集会尝试以下候选路径：

```text
{data_path}/instances_val2017.json
{data_path}/annotations/instances_val2017.json
{data_path}/annotations/instances_val.json
{data_path}/annotations/instances_{split}.json
{data_path}/val/_annotations.coco.json
{data_path}/instances_val.json
{data_path}/_annotations.coco.json
```

这可以兼容常见 COCO 导出结构。

### 5.3 image_id 映射

COCO predictions JSON 中的 `image_id` 必须与 annotation JSON 中的 `images[].id` 一致。当前实现会在初始化指标时读取 annotation JSON，并建立：

```python
self.img_id_map[Path(img["file_name"]).name] = img["id"]
self.img_id_map[Path(img["file_name"]).stem] = img["id"]
```

预测写入 JSON 时优先使用该映射，只有找不到时才回退到原始文件名转换逻辑。

这个改动解决了自定义图片文件名无法可靠转换为整数 ID 的问题。

### 5.4 COCOeval 指标写回

`coco_evaluate()` 完成以下步骤：

1. 加载 annotation JSON。
2. 加载 predictions JSON。
3. 根据验证集图片列表设置 `val.params.imgIds`。
4. 调用 `evaluate()`、`accumulate()`、`summarize()`。
5. 将结果写回 stats。

主要指标包括：

```python
stats["metrics/mAP50(B)"] = val.stats[1]
stats["metrics/mAP50-95(B)"] = val.stats[0]
stats["metrics/mAP_small(B)"] = val.stats[3]
stats["metrics/mAP_medium(B)"] = val.stats[4]
stats["metrics/mAP_large(B)"] = val.stats[5]
```

训练期最终使用 `metrics/mAP50-95(B)` 作为主要 fitness。

## 6. 模型 YAML 注释

以下文件补充了结构注释，帮助阅读模块参数和 residual 设置：

1. `ultralytics/cfg/models/v8/yolov8.yaml`
2. `ultralytics/cfg/models/v10/yolov10s.yaml`
3. `ultralytics/cfg/models/12/yolo12.yaml`
4. `ultralytics/cfg/models/26/yolo26.yaml`

这些注释不改变模型结构。

## 7. AMP 检查模型回退

文件：`ultralytics/utils/checks.py`

AMP 检查中的示例模型从 `yolo26n.pt` 改为更常见的 `yolov8n.pt`，以提高环境兼容性。

## 8. 示例脚本与效果图

本次开源包含以下脚本：

| 文件 | 说明 |
| --- | --- |
| `ultralytics/train.py` | 训练入口示例，展示 COCO fitness 参数。 |
| `coco-test.py` | COCO 相关测试脚本。 |
| `visual.py` | 可视化脚本。 |
| `ultralytics/plotfig2.py` | 绘图脚本。 |
| `ultralytics/3d.py` | 3D/可视化相关脚本。 |

效果图：

1. `ultralytics/example1.jpg`
2. `ultralytics/example2.jpg`

## 9. 当前训练流程总结

启用 COCO fitness 后，训练流程可以概括为：

1. 训练器读取并补全 COCO fitness 参数。
2. 每个 epoch 验证前判断是否需要运行 COCO API。
3. 不需要运行时跳过 JSON 收集，降低开销。
4. 需要运行时收集预测并保存 predictions JSON。
5. 检测验证器查找 annotation JSON 并建立 image_id 映射。
6. `pycocotools.COCOeval` 计算 COCO 指标。
7. `mAP50-95(B)` 写回 fitness。
8. 训练器根据 `coco_eval` 标记决定是否更新 `best.pt`。

## 10. 使用建议

严格按 COCO 指标选择模型时，建议：

```python
save_json=True
use_coco_fitness=True
coco_eval_interval=5或10
coco_only_best=True
coco_start_epoch=适当的热身轮数
```

快速调试训练流程时，建议：

```python
save_json=False
use_coco_fitness=False
```

使用自定义 COCO 数据集时，应确认 annotation JSON 中的 `images[].file_name` 能与验证图片文件名对应。
