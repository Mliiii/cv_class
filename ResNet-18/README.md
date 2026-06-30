# CIFAR-10 ResNet-18 变体训练作业

本项目对应计算机视觉课程必做任务 01：使用 CIFAR-10 数据集，从零实现并训练一个 ResNet-18 变体模型，目标是在验证集上取得超过 85% 的分类准确率。

## 目录结构

```text
ResNet-18/
├── README.md
├── requirements.txt
└── train.py
```

## 实现内容

- 从零实现 ResNet-18 变体，没有使用 torchvision 自带的预训练模型。
- 针对 CIFAR-10 的 32x32 小图像修改网络入口：使用 `3x3` 卷积，步长为 1，并移除 ImageNet 版 ResNet 的 `7x7` 卷积和最大池化层。
- 使用 `BasicBlock` 残差块，网络层数配置为 `[2, 2, 2, 2]`。
- 训练策略包含 SGD + Momentum、Nesterov、weight decay、warmup + cosine 学习率衰减，并提供 label smoothing、RandAugment、Random Erasing 等可选增强。
- 默认数据增强包含随机裁剪和随机水平翻转；RandAugment 和 Random Erasing 可通过参数开启。
- 自动保存 `last.pt`、`best.pt`、`metrics.csv` 和 `config.json`。

## 环境安装

建议使用 Python 3.9+，并安装 PyTorch 与 torchvision：

```bash
pip install -r requirements.txt
```

如果机器支持 CUDA，请安装与本机 CUDA 版本匹配的 PyTorch。

## 数据集路径

当前代码默认使用已经解压好的 CIFAR-10 数据集：

```text
/datasets/cifar-10-batches-py
```

`torchvision.datasets.CIFAR10` 实际需要传入 `cifar-10-batches-py` 的父目录。代码已经做了自动转换，所以以下两种写法都可以：

```bash
--data-dir /datasets/cifar-10-batches-py
--data-dir /datasets
```

## 开始训练

进入当前目录：

```bash
cd /workspace/project/Computer_Vision/ResNet-18
```

推荐训练命令，会按默认 200 轮正常训练，85% 作为最终验收指标：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train.py --amp
```

如果需要显式指定数据路径：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train.py \
  --data-dir /datasets/cifar-10-batches-py \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --amp
```

## 输出文件

默认输出目录为：

```text
./runs/resnet18_cifar10
```

其中：

- `best.pt`：验证集准确率最高的模型权重。
- `last.pt`：最后一个 epoch 的模型权重。
- `metrics.csv`：每轮训练的 loss、accuracy、learning rate 和耗时。
- `config.json`：本次训练使用的参数配置。

## 复现实验配置

推荐配置如下：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train.py \
  --data-dir /datasets/cifar-10-batches-py \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --amp
```

在常见 GPU 环境下，CIFAR-10 版 ResNet-18 使用随机裁剪、随机水平翻转、SGD + cosine 学习率策略，通常可以超过 85% 验证集精度。若希望进一步提高最终精度，可以完整训练 200 轮，或尝试开启 `--label-smoothing 0.1 --randaugment --random-erasing 0.25`。

## 方法说明

原始 ResNet-18 面向 ImageNet，输入图像尺寸较大，因此第一层使用 `7x7` 大卷积和最大池化。CIFAR-10 图像只有 `32x32`，如果仍使用原始入口，会过早丢失空间细节。本作业中的变体将入口改为 `3x3` 卷积，并保留更多早期特征，有利于小图像分类。

训练时采用带动量的 SGD 优化器，并使用 cosine 学习率衰减。前若干 epoch 使用 warmup，避免训练初期学习率过大导致不稳定。label smoothing 可以降低模型过度自信，RandomCrop、RandomHorizontalFlip、RandAugment 和 Random Erasing 用于提升泛化能力。

## 结果记录模板

训练完成后，可以在实验报告中填写：

| 项目 | 内容 |
| --- | --- |
| 数据集 | CIFAR-10 |
| 模型 | CIFAR-10 ResNet-18 variant |
| 优化器 | SGD + Momentum + Nesterov |
| 学习率策略 | Warmup + CosineAnnealing |
| 损失函数 | CrossEntropyLoss，label smoothing 可选 |
| 数据增强 | 默认 RandomCrop、RandomHorizontalFlip；可选 RandAugment、RandomErasing |
| 最佳验证准确率 | 查看 `runs/resnet18_cifar10/metrics.csv` 中的 `best_acc` |

## 断点续训

如果训练中断，可以从最近一次权重继续：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train.py --resume runs/resnet18_cifar10/last.pt --amp
```

