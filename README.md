# LDE-Net

### A lightweight dynamic kernel and edge-guided network for mammographic mass detection

[English](README.md) | [简体中文](README.zh-CN.md)

Implementation accompanying the manuscript by **Chang Yang, Jing Peng, Wei Yao, and Shengzhou Xu**.

LDE-Net detects breast masses in mammograms. Built on YOLO11, it combines adaptive feature extraction, multiscale edge guidance, branch-specific attention, and geometric bounding-box regression to handle variations in lesion size, shape, and boundary contrast.

## Method

- **Dynamic inception mixer block (DIMB):** input-dependent routing across square, horizontal, and vertical depthwise convolution branches.
- **Edge information transfer module (EITM):** Sobel edge extraction and multiscale max pooling transfer shallow boundary information to deeper features.
- **Separation enhancement attention head (SEAHead):** separate attention modules recalibrate the classification and regression branches.
- **FMIoU loss:** overlap remapping and a two-corner distance penalty provide bounding-box supervision. The default interval is `d = 0.0`, `u = 0.95`.

## Results reported in the manuscript

On the combined **DDSM/MIAS** test set, LDE-Net achieved **86.9% precision, 79.1% recall, 82.8% F1, 85.8% mAP@50, and 48.5% mAP@50:95**, with **2.6 million parameters** and **6.6 GFLOPs**. The reported inference speed was **247.4 frames/s** on an NVIDIA GeForce RTX 4070 SUPER.

On the independent **INbreast** test set of **103 images with 106 annotated lesions**, it achieved **89.7% precision, 80.9% recall, 85.1% F1, 86.6% mAP@50, and 39.2% mAP@50:95**.

## Code organization

The main model configuration is:

[`ultralytics/cfg/models/end/MODULES/yolo11_DIMB_GEIT_SEAHead_FMIOU.yaml`](ultralytics/cfg/models/end/MODULES/yolo11_DIMB_GEIT_SEAHead_FMIOU.yaml)

The implementation retains the original development names:

- **DIMB:** `DynamicInceptionDWConv2d`, `DynamicInceptionMixer`, `DynamicIncMixerBlock`, and `C3k2_DCMB` in [`block.py`](ultralytics/nn/extra_modules/block.py).
- **EITM:** `SobelConv`, `MutilScaleEdgeInfoGenetator`, and `ConvEdgeFusion` in [`block.py`](ultralytics/nn/extra_modules/block.py).
- **SEAHead:** `Detect_SEAM` in [`head.py`](ultralytics/nn/extra_modules/head.py), with `SEAM` in [`block.py`](ultralytics/nn/extra_modules/block.py).
- **FMIoU:** `bbox_focaler_mpdiou` in [`metrics.py`](ultralytics/utils/metrics.py), called by `BboxLoss` in [`loss.py`](ultralytics/utils/loss.py).

Component configurations are under [`MODULES`](ultralytics/cfg/models/end/MODULES); loss and pooling comparisons are under [`LOSS`](ultralytics/cfg/models/end/LOSS) and [`AVG_MAX`](ultralytics/cfg/models/end/AVG_MAX). Loss selection is implemented in `BboxLoss`; selecting a differently named YAML file alone does not switch the loss. The current code uses FMIoU with `self.use_wiseiou = False`.

## Environment and installation

The manuscript experiments used **Python 3.10.14, PyTorch 2.2.2, torchvision 0.17.2, and CUDA 12.1**. The bundled Ultralytics code is based on version **8.3.9**.

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

The PyTorch command follows the [official installation instructions for previous versions](https://pytorch.org/get-started/previous-versions/). Install this checkout in editable mode so that Python loads its custom modules.

The repository also contains experimental backbones and operators. Their additional dependencies and compiled CUDA extensions depend on the modules being imported. These are not all declared in `pyproject.toml`; use the build instructions supplied with the relevant extension when needed.

Check the main model from the repository root:

```python
from ultralytics import YOLO

model = YOLO(
    "ultralytics/cfg/models/end/MODULES/yolo11_DIMB_GEIT_SEAHead_FMIOU.yaml"
)
model.info()
```

## Dataset preparation

Prepare one detection class, `tumor`, in YOLO format. Each image has a matching text file containing one row per annotated mass:

```text
class_id x_center y_center width height
```

Use class ID `0` and normalize all four coordinates by image width or height. A typical layout is:

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

Create `data.yaml` with paths to your prepared dataset:

```yaml
path: /absolute/path/to/breast_dataset
train: images/train
val: images/val
test: images/test
nc: 1
names: [tumor]
```

The existing [`b_data.yaml`](ultralytics/cfg/datasets/b_data.yaml) and [`a_data.yaml`](ultralytics/cfg/datasets/a_data.yaml) refer to DDSM/MIAS and INbreast, respectively; update their local paths before use. Add a `test` entry for held-out evaluation.

The manuscript uses a patient-level split: training plus validation versus testing at **8:2**, followed by training versus validation at **8:2** within the development pool (overall **64:16:20**). INbreast is reserved for external testing. Obtain the datasets from their respective providers and prepare the partitions before training.

## Training

Save the following as `train_lde.py` in the repository root, then run `python train_lde.py`. It selects the full LDE-Net architecture and trains from scratch. The existing root-level scripts are experiment entry points with their own model and data paths.

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

This example specifies the architecture and principal training settings. Augmentation and other options are read from [`default.yaml`](ultralytics/cfg/default.yaml) unless overridden in `model.train(...)`. Set these explicitly for the experiment you intend to run.

Training writes checkpoints and logs to `runs/ldenet/train/` (or an incremented directory if it already exists). `runs/` is excluded from version control.

## Test-set mAP evaluation

Use a trained LDE-Net checkpoint and a dataset YAML containing a `test` split:

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

The low confidence cutoff retains detections for the precision-recall curve used by mAP. For fixed-threshold precision, recall, and F1, the manuscript selects a confidence threshold on the source validation set and holds it fixed for testing. The built-in validator's summary precision and recall are taken at its best-F1 operating point; they are separate from that fixed-threshold evaluation.

## Inference

Set `confidence_threshold` to the operating threshold selected on your validation set:

```python
from ultralytics import YOLO

if __name__ == "__main__":
    confidence_threshold = 0.25  # Example value; choose on the validation set.
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

Training checkpoints from `runs/` are not included in this repository. The root-level `yolo11n.pt` and `yolov8n.pt` files are baseline model weights, not trained LDE-Net checkpoints.

## Citation

```bibtex
@misc{yang2026ldenet,
  title  = {{LDE-Net}: A lightweight dynamic kernel and edge-guided network for mammographic mass detection},
  author = {Yang, Chang and Peng, Jing and Yao, Wei and Xu, Shengzhou},
  year   = {2026},
  url    = {https://github.com/myhhkb/LDENet}
}
```

## Acknowledgments and license

This project builds on [Ultralytics](https://github.com/ultralytics/ultralytics) and incorporates modules from the research implementations credited in the source files. We thank their authors for making their work available.

The repository retains the upstream [AGPL-3.0 license](LICENSE). Third-party components retain their respective notices and license terms.

## Contact

- **Chang Yang:** [2024120443@mail.scuec.edu.cn](mailto:2024120443@mail.scuec.edu.cn)
- **Shengzhou Xu (corresponding author):** [xushengzhou@scuec.edu.cn](mailto:xushengzhou@scuec.edu.cn)
