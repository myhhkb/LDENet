import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples

if __name__ == '__main__':
    model = YOLO('runs/xiaorong_module/GEIT_DIMB_SEAM_FMIOU/yolo11/weights/best.pt')
    model.export(format='onnx', simplify=True, opset=13)