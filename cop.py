import os
import shutil

# 你已有的文件路径列表
file_paths = [
    "ultralytics/cfg/models/11/yolo11-C3k2-MutilScaleEdgeInformationEnhance.yaml",
    "ultralytics/cfg/models/11/yolo11-C3k2-EMSC.yaml",
    "ultralytics/cfg/models/11/yolo11-C3k2-MutilScaleEdgeInformationSelect.yaml",
    "ultralytics/cfg/models/11/yolo11-C3k2-MutilScaleEdgeInformationEnhance.yaml",
    "ultralytics/cfg/models/11/yolo11-HAFB-1.yaml",
    "ultralytics/cfg/models/11/yolo11-HAFB-2.yaml",
    "ultralytics/cfg/models/11/yolo11-MutilBackbone-DAF.yaml",
    "ultralytics/cfg/models/11/yolo11-CSP-FreqSpatial.yaml",
    "ultralytics/cfg/models/11/yolo11-CSP-PMSFA.yaml",
    "ultralytics/cfg/models/11/yolo11-EMBSFPN.yaml",
    "ultralytics/cfg/models/11/yolo11-FeaturePyramidSharedConv.yaml",
    # "ultralytics/cfg/models/11/yolo11-CGRFPN.yaml",
    # "ultralytics/cfg/models/11/yolo11-SOEP.yaml",
    "ultralytics/cfg/models/11/yolo11-CSP-PTB.yaml",
    "ultralytics/cfg/models/11/yolo11-C3k2-SMPCGLU.yaml",
    # "ultralytics/cfg/models/11/yolo11-LSDECD.yaml",
    "ultralytics/cfg/models/11/yolo11-ContextGuideFPN.yaml",
    "ultralytics/cfg/models/11/yolo11-C3k2-EIEM.yaml",
    "ultralytics/cfg/models/11/yolo11-EIEStem.yaml",
    "ultralytics/cfg/models/11/yolo11-LSCSBD.yaml",
    "ultralytics/cfg/models/11/yolo11-RGCSPELAN.yaml",
    "ultralytics/cfg/models/11/yolo11-FDPN-DASI.yaml",
    "ultralytics/cfg/models/11/yolo11-FDPN.yaml",
    "ultralytics/cfg/models/11/yolo11-TADDH.yaml",
    "ultralytics/cfg/models/11/yolo11-LSCD.yaml",
    "ultralytics/cfg/models/11/yolo11-C3k2-EMSCP.yaml",
    "ultralytics/cfg/models/11/yolo11-C3k2-EMSC.yaml",
    # 在这里继续加 ...
]

# 目标目录
target_dir = "ultralytics/cfg/models/our"

# 若目标路径不存在则自动创建
os.makedirs(target_dir, exist_ok=True)

for f in file_paths:
    if os.path.isfile(f):
        shutil.copy(f, target_dir)
        print(f"Copied: {f} -> {target_dir}")
    else:
        print(f"File not found: {f}")
