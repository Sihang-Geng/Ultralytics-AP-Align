import argparse
import json
import os
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(pred_json, anno_json):
    print(f"\n正在评估 COCO 指标，使用 {pred_json} 和 {anno_json}...")
    
    # 检查文件是否存在
    for x in [pred_json, anno_json]:
        assert os.path.isfile(x), f"文件 {x} 不存在"
    
    # 初始化COCO API
    anno = COCO(str(anno_json))  # 初始化标注API
    pred = anno.loadRes(str(pred_json))  # 初始化预测API (必须传递字符串，而非Path对象)
    
    # 进行bbox评估
    eval_bbox = COCOeval(anno, pred, 'bbox')
    eval_bbox.evaluate()
    eval_bbox.accumulate()
    eval_bbox.summarize()


def main():

    annotations_path = "instances_val2017.json"


    predictions_path = "/root/ultralytics/runs/detect/val2/predictions.json"
    # ===========================================
    print(f"使用标注文件: {annotations_path}")
    print(f"使用预测文件: {predictions_path}")
    
    # 确保文件路径存在
    pred_json = Path(predictions_path)
    anno_json = Path(annotations_path)

   
    # 评估并打印结果
    stats = evaluate_coco(pred_json, anno_json)


if __name__ == '__main__':
    main() 
