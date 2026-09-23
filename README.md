# 图像分类模型对比：ResNet、MobileNetV3、ViT

面向初学者的 PyTorch 图像分类实验项目。用统一的数据处理、训练循环和评估方式，对比 ResNet-18/34、MobileNetV3-Small、ViT-Tiny 在 CIFAR-10 和 Tiny ImageNet 上的准确率、参数量、计算量、训练时间与训练轮数。

> 默认配置用于入门和快速复现。不同硬件、随机种子、训练轮数会影响结果；比较时请以本次生成的 `results/` 文件为准。FLOPs 按每张输入图像估算，口径为 MACs×2。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python train.py --dataset cifar10 --model resnet18 --epochs 5
```

模型选项：`resnet18`、`resnet34`、`mobilenet_v3_small`、`vit_tiny`。数据集选项：`cifar10`、`tiny_imagenet`。首次运行会将数据下载到 `data/`。Tiny ImageNet 约 100,000 张训练图像，首次下载和解压需要较多磁盘空间。

## 运行完整对比

```bash
python compare.py --epochs 5
```

每个模型依次在两个数据集上训练。可用 `--models resnet18 mobilenet_v3_small` 或 `--datasets cifar10` 缩小实验。默认种子为 42；可用 `--device auto|cpu|cuda|mps` 选择设备。

## 项目结构

```text
models/       模型定义
data/         自动下载的数据集（不提交）
checkpoints/  最佳验证准确率权重（Git 忽略，随 Release 发布）
results/      训练日志、指标表和对比图（提交轻量结果文件）
 train.py     单次训练入口
 compare.py   批量实验入口
```

每次训练会保存 `checkpoints/<dataset>/<model>.pt`（模型权重及配置）、`results/logs/` 下的逐轮 CSV 和 JSON 指标，以及 `results/figures/` 下的损失/准确率曲线。完整对比结束后会输出汇总表和准确率、参数/FLOPs、训练耗时对比图。

## 说明

- CIFAR-10：60,000 张 32×32 彩色图像，10 类。Tiny ImageNet：约 100,000 张训练图像、10,000 张验证图像，200 类、64×64。模型统一缩放至 224×224，以兼容 ViT 输入。
- 默认训练采用 AdamW 与交叉熵，按验证准确率保存最佳权重。所有模型从头训练，不加载预训练权重，以保持实验条件清晰。
- 参数量和计算量是模型结构的静态估算；训练耗时受设备影响。比较实验建议固定设备、随机种子和轮数。
- 权重文件较大，Git 默认忽略 `checkpoints/` 和数据，避免把大型二进制文件写入仓库历史。训练日志与图表保存在 `results/`。发布权重时，将 `checkpoints/` 压缩后附加到 GitHub Release。
