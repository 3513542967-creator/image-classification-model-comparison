# 图像分类模型对比

面向初学者的 PyTorch 项目：从头训练 ResNet-18、ResNet-34、MobileNetV3-Small 和 ViT-Tiny，在 CIFAR-10 与 Tiny ImageNet 上比较准确率、参数量、计算量和收敛时间。

## 训练

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_full.py
```

程序会下载数据、依次训练 8 组实验，并在中断后继续未完成的训练。统一使用 64×64 输入、随机种子 42；最多训练 40 轮，验证准确率连续 8 轮不提高时提前停止。

## 本次结果

| 模型 | CIFAR-10 测试准确率 | Tiny ImageNet 测试准确率 |
| --- | ---: | ---: |
| ResNet-18 | 85.21% | **39.00%** |
| ResNet-34 | **86.10%** | 37.99% |
| MobileNetV3-Small | 76.35% | 32.83% |
| ViT-Tiny | 74.40% | 30.84% |

- [完整结果与实验说明](RESULTS_FULL.md)：参数量、FLOPs、训练轮数、耗时及结果解读。
- [逐轮日志和对比图](results/full/) · [汇总 CSV](results/full/comparison.csv)
- [下载 8 份最佳权重](https://github.com/3513542967-creator/image-classification-model-comparison/releases/tag/v0.2.0)

权重包解压到项目根目录后，可直接加载，例如：

```python
import torch
from models import build_model

model = build_model("resnet18", num_classes=10, image_size=64)
model.load_state_dict(torch.load(
    "checkpoints/full/cifar10/resnet18/best.pt",
    map_location="cpu", weights_only=True,
))
model.eval()
```

`train.py` 和 `compare.py` 是首版快速演示脚本；正式实验请运行 `run_full.py`。本仓库不包含数据集文件，模型权重随 Release 发布。
