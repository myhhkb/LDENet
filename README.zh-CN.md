# LDE-Net

### 用于乳腺 X 线肿块检测的轻量级动态卷积核与边缘引导网络

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库为论文 **LDE-Net: A lightweight dynamic kernel and edge-guided network for mammographic mass detection** 的配套代码。

**作者：杨畅、彭婧、姚为、徐胜舟。**

LDE-Net 基于 YOLO11，通过动态特征提取、多尺度边缘传递、分类与回归分支注意力以及几何回归损失，实现乳腺 X 线图像中的肿块检测。

## 方法组成

- **DIMB（动态 inception 混合模块）：** 根据输入特征，动态融合方形、水平和垂直深度卷积分支。
- **EITM（边缘信息传递模块）：** 利用 Sobel 算子和多尺度最大池化，将浅层边界信息传递至深层特征。
- **SEAHead（分离增强注意力检测头）：** 分别对分类和回归分支进行特征重标定。
- **FMIoU 损失：** 结合重叠度重映射和双角点距离惩罚，默认参数为 `d=0.0`、`u=0.95`。

## 论文结果

在 **DDSM/MIAS** 测试集上，Precision、Recall、F1、mAP@50 和 mAP@50:95 分别为 **86.9%、79.1%、82.8%、85.8% 和 48.5%**。模型参数量为 **2.6 M**，计算量为 **6.6 GFLOPs**；在 NVIDIA GeForce RTX 4070 SUPER 上报告的推理速度为 **247.4 frames/s**。

在独立的 **INbreast** 测试集（**103 幅图像、106 个标注病灶**）上，上述五项指标分别为 **89.7%、80.9%、85.1%、86.6% 和 39.2%**。

## 核心代码

完整模型配置：

[`ultralytics/cfg/models/end/MODULES/yolo11_DIMB_GEIT_SEAHead_FMIOU.yaml`](ultralytics/cfg/models/end/MODULES/yolo11_DIMB_GEIT_SEAHead_FMIOU.yaml)

源码保留了开发阶段的类名，对应关系如下：

- **DIMB：** [`block.py`](ultralytics/nn/extra_modules/block.py) 中的 `DynamicInceptionDWConv2d`、`DynamicInceptionMixer`、`DynamicIncMixerBlock` 和 `C3k2_DCMB`。
- **EITM：** 同文件中的 `SobelConv`、`MutilScaleEdgeInfoGenetator` 和 `ConvEdgeFusion`。
- **SEAHead：** [`head.py`](ultralytics/nn/extra_modules/head.py) 中的 `Detect_SEAM`，以及 `block.py` 中的 `SEAM`。
- **FMIoU：** [`metrics.py`](ultralytics/utils/metrics.py) 中的 `bbox_focaler_mpdiou`，由 [`loss.py`](ultralytics/utils/loss.py) 中的 `BboxLoss` 调用。

模块、损失函数和池化方式的对比配置分别位于 [`MODULES`](ultralytics/cfg/models/end/MODULES)、[`LOSS`](ultralytics/cfg/models/end/LOSS) 和 [`AVG_MAX`](ultralytics/cfg/models/end/AVG_MAX)。损失函数由 `BboxLoss` 中的实现决定，单独更换 YAML 文件名不会切换损失。当前代码已通过 `self.use_wiseiou = False` 启用 FMIoU。

## 环境与安装

论文实验环境为 **Python 3.10.14、PyTorch 2.2.2、torchvision 0.17.2、CUDA 12.1**。仓库内 Ultralytics 代码基于 **8.3.9** 版本。

```bash
git clone https://github.com/myhhkb/LDENet.git
cd LDENet
conda create -n ldenet python=3.10.14 -y
conda activate ldenet
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install "numpy<2" "opencv-python<4.12"
python -m pip install timm==1.0.7 einops==0.8.1 efficientnet-pytorch==0.7.1 pytorch-wavelets==1.3.0 PyWavelets==1.8.0
python -m pip install -e .
```

PyTorch 安装命令参考[官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。使用可编辑安装加载本仓库中的自定义模块。

仓库还保留了其他实验骨干网络与算子。根据实际导入的模块，可能需要安装对应依赖或编译 CUDA 扩展；这些扩展并未全部列入 `pyproject.toml`，安装时参考相应模块附带的构建说明。

在项目根目录检查模型：

```python
from ultralytics import YOLO

model = YOLO(
    "ultralytics/cfg/models/end/MODULES/yolo11_DIMB_GEIT_SEAHead_FMIOU.yaml"
)
model.info()
```

## 数据准备

使用 YOLO 检测标注格式，类别为 `tumor`，类别编号为 `0`。每幅图像对应一个同名 `.txt` 文件，每个肿块占一行：

```text
class_id x_center y_center width height
```

中心坐标和宽高均按图像尺寸归一化。目录示例：

```text
breast_dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

创建 `data.yaml`：

```yaml
path: /absolute/path/to/breast_dataset
train: images/train
val: images/val
test: images/test
nc: 1
names: [tumor]
```

现有 [`b_data.yaml`](ultralytics/cfg/datasets/b_data.yaml) 和 [`a_data.yaml`](ultralytics/cfg/datasets/a_data.yaml) 分别对应 DDSM/MIAS 和 INbreast，使用前需修改为自己的数据路径，并为测试集增加 `test` 字段。

论文按患者划分数据：先按 **训练集与验证集合计∶测试集 = 8∶2** 划分，再将前者按 **训练集∶验证集 = 8∶2** 划分，总体比例约为 **64∶16∶20**。INbreast 仅用于独立外部测试。请从各数据集提供方获取数据，并在训练前完成划分。

## 训练

将以下代码保存为项目根目录下的 `train_lde.py`，运行 `python train_lde.py`。该示例指定完整 LDE-Net 配置，并从头训练。仓库根目录现有脚本为不同实验的入口，各自带有模型和数据路径。

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(
        "ultralytics/cfg/models/end/MODULES/yolo11_DIMB_GEIT_SEAHead_FMIOU.yaml"
    )
    model.train(
        data="data.yaml",
        imgsz=640,
        epochs=300,
        batch=8,
        optimizer="SGD",
        lr0=0.01,
        weight_decay=0.0005,
        pretrained=False,
        resume=False,
        patience=0,
        device=0,
        workers=0,
        amp=False,
        project="runs/ldenet",
        name="train",
    )
```

示例设置了网络结构和主要训练参数。数据增强及其他选项默认读取 [`default.yaml`](ultralytics/cfg/default.yaml)，可通过 `model.train(...)` 显式覆盖，按所运行的实验设置。

权重和日志保存在 `runs/ldenet/train/` 中；已有同名目录时会自动递增目录名。`runs/` 已排除在 Git 版本管理之外。

## 测试集 mAP 评估

使用训练后的 LDE-Net 权重，并在数据 YAML 中配置 `test` 路径：

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/ldenet/train/weights/best.pt")
    metrics = model.val(
        data="data.yaml",
        split="test",
        imgsz=640,
        batch=1,
        conf=0.001,
        iou=0.70,
        rect=False,
        device=0,
        project="runs/ldenet",
        name="test",
    )
    print("mAP@50:", metrics.box.map50)
    print("mAP@50:95:", metrics.box.map)
```

较低的置信度截断值用于保留构建 mAP 精确率–召回率曲线所需的检测结果。论文中的固定阈值 Precision、Recall 和 F1 使用源域验证集选出的阈值，并将其固定用于测试。内置验证器汇总的 Precision 和 Recall 取自其最佳 F1 工作点，与固定阈值评估需要区分。

## 推理

将 `confidence_threshold` 设置为验证集选定的工作阈值：

```python
from ultralytics import YOLO

if __name__ == "__main__":
    confidence_threshold = 0.25  # 示例值，实际使用验证集选定的阈值。
    model = YOLO("runs/ldenet/train/weights/best.pt")
    model.predict(
        source="path/to/mammograms",
        imgsz=640,
        conf=confidence_threshold,
        iou=0.70,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/ldenet",
        name="predict",
    )
```

仓库不包含 `runs/` 中的训练权重。根目录中的 `yolo11n.pt` 和 `yolov8n.pt` 是基线模型权重，不是训练后的 LDE-Net 权重。

## 引用

```bibtex
@misc{yang2026ldenet,
  title  = {{LDE-Net}: A lightweight dynamic kernel and edge-guided network for mammographic mass detection},
  author = {Yang, Chang and Peng, Jing and Yao, Wei and Xu, Shengzhou},
  year   = {2026},
  url    = {https://github.com/myhhkb/LDENet}
}
```

## 致谢与许可证

本项目基于 [Ultralytics](https://github.com/ultralytics/ultralytics)，并使用了源码中注明出处的研究模块，感谢相关作者开放其实现。

仓库保留上游 [AGPL-3.0 许可证](LICENSE)。第三方组件保留各自的版权声明与许可条款。

## 联系方式

- **杨畅：** [2024120443@mail.scuec.edu.cn](mailto:2024120443@mail.scuec.edu.cn)
- **徐胜舟（通讯作者）：** [xushengzhou@scuec.edu.cn](mailto:xushengzhou@scuec.edu.cn)
