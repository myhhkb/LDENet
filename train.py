import warnings, os,sys
warnings.filterwarnings('ignore')
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml')
model.train(data='ultralytics/cfg/datasets/a_data.yaml',
                imgsz=640,
                epochs=300,
                batch=8,
                workers=0, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                device='0', # 指定显卡和多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
                optimizer='SGD', # using SGD
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,不懂就在百度云.txt找断点续训的视频
                amp=False, # close amp | loss出现nan可以关闭amp
                project='runs/xiaorong_module',
                pretrained= True,
                name="MBM-YOLO-INBREAST",
                )

