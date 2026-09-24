import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/xiaorong_module/yolo11_GEIT_SEAHead/weights/best.pt') # select your model.pt path
    model.predict(source='heatmap/INbreast_640/val/images',
                  imgsz=640,
                  project='runs/detect_False/yolo11_GEIT_SEAHead',
                  name='INbreast_640',
                  save=True,
                  conf=0.2,
                  iou=0.5,
                  agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  line_width=1, # line width of the bounding boxes
                  show_conf=False, # do not show prediction confidence
                  show_labels=False, # do not show prediction labels
                  save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )