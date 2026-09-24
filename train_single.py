# train_single.py
import sys, os
from ultralytics import YOLO

cfg_path = sys.argv[1]  # 从命令行接收yaml路径
model_name = os.path.splitext(os.path.basename(cfg_path))[0]

print(f"\n===== 开始训练模型：{model_name} =====\n")

model = YOLO(cfg_path)
model.train(
    data='ultralytics/cfg/datasets/b_data.yaml',
    imgsz=640,
    epochs=300,
    batch=8,
    workers=0,
    device='0',
    optimizer='SGD',
    project='runs/xiaorong_loss',
    name=model_name,
    amp=False,
    patience=0, # set 0 to close earlystop.
    resume=True, # 断点续训,YOLO初始化时选择last.pt,不懂就在百度云.txt找断点续训的视频
    # fraction=0.2,
)
