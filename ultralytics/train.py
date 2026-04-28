from ultralytics import YOLO

#cd ultralytics/ultralytics
#python xuxun.py
#  nohup python 训练.py > training.log 2>&1 &
model = YOLO(
    "/root/ultralytics/ultralytics/cfg/models/v8/yolov8s.yaml"
  #  task="detect",
)
#cd ultralytics-main/ultralytics
results = model.train(
    data="/root/ultralytics/ultralytics/cfg/真的RUOD/RUOD_YOLO/data.yaml", 
    epochs=250, 
    imgsz=640,  
    seed=0, 
    deterministic=True,
    save_json=True,
    use_coco_fitness=True,
    coco_eval_interval=10,
    coco_only_best=True,
    coco_start_epoch=100,
    patience=100
)
# - save_json=True ：生成 COCO 评估需要的预测 JSON。
# - use_coco_fitness=True ：启用 COCO API 评估并用 COCO mAP 作为 fitness。
# - coco_eval_interval=1 ：每个 epoch 都跑 COCO；如果想降低开销，改成 5、10 等即可。
# - coco_start_epoch=100 : 前100轮不进行COCO评估（也不保存最佳模型），加速前期训练。
results = model.val()


