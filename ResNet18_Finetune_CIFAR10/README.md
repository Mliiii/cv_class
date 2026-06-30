# ImageNet 预训练 ResNet-18 微调 CIFAR-10

本文件夹对应计算机视觉课程作业：加载 PyTorch 官方提供的 ImageNet 预训练 ResNet-18 权重，在 CIFAR-10 上进行微调，并从最终分类精度和模型收敛速度两个维度对比“从零训练”和“迁移学习微调”的效果差异。

## 目录结构

```text
ResNet18_Finetune_CIFAR10/
├── README.md
├── compare_runs.py
├── requirements.txt
└── train_finetune.py
```

## 数据集路径

当前代码默认读取：

```text
/datasets/cifar-10-batches-py
```

`torchvision.datasets.CIFAR10` 需要传入 `cifar-10-batches-py` 的父目录。脚本已自动兼容，所以以下两种写法都可以：

```bash
--data-dir /datasets/cifar-10-batches-py
--data-dir /datasets
```

## 训练微调模型

进入作业目录：

```bash
cd /workspace/project/Computer_Vision/ResNet18_Finetune_CIFAR10
```

推荐运行：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train_finetune.py --amp
```

默认配置会加载 `torchvision.models.resnet18` 的 ImageNet 预训练权重，将最后的 `fc` 层替换为 CIFAR-10 的 10 类分类头，先冻结 backbone 训练 1 个 epoch，然后解冻全部网络进行微调。默认不会因为达到 85% 而提前停止，会按设定 epoch 正常训练。

如果当前机器没有缓存官方 ResNet-18 权重且不能联网下载，可以先把官方权重文件放到本地，然后使用：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train_finetune.py \
  --weights-path /path/to/resnet18-f37072fd.pth \
  --amp
```

快速检查代码和数据路径：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train_finetune.py \
  --no-pretrained \
  --epochs 0 \
  --device cpu \
  --workers 0
```

## 输出文件

默认输出目录：

```text
runs/resnet18_finetune_cifar10
```

主要文件：

- `best.pt`：验证集精度最高的权重。
- `last.pt`：最后一轮权重。
- `metrics.csv`：每轮训练 loss、accuracy、学习率和耗时。
- `config.json`：训练参数。

## 与从零训练对比

先分别完成两个实验：

```bash
cd /workspace/project/Computer_Vision/ResNet-18
/workspace/miniconda3/envs/downtime66/bin/python train.py --amp

cd /workspace/project/Computer_Vision/ResNet18_Finetune_CIFAR10
/workspace/miniconda3/envs/downtime66/bin/python train_finetune.py --amp
```

然后运行对比脚本：

```bash
cd /workspace/project/Computer_Vision/ResNet18_Finetune_CIFAR10
/workspace/miniconda3/envs/downtime66/bin/python compare_runs.py
```

如果日志路径不同，可以手动指定：

```bash
/workspace/miniconda3/envs/downtime66/bin/python compare_runs.py \
  --scratch-metrics /workspace/project/Computer_Vision/ResNet-18/runs/resnet18_cifar10/metrics.csv \
  --finetune-metrics runs/resnet18_finetune_cifar10/metrics.csv
```

对比结果会保存到：

```text
runs/comparison.md
runs/comparison.json
```

## 分析思路

可以从两个维度写实验分析：

| 维度 | 观察指标 | 说明 |
| --- | --- | --- |
| 最终分类精度 | `best_acc` 或最高 `val_acc` | 反映模型最终泛化能力 |
| 收敛速度 | 达到 85% 精度所需 epoch | epoch 越少，说明收敛越快 |

通常情况下，ImageNet 预训练 ResNet-18 在 CIFAR-10 上微调会比从零训练更快达到较高精度，因为预训练模型已经学习了通用视觉特征；从零训练需要先学习底层边缘、纹理和形状特征，前期收敛较慢，但训练充分后也可能获得接近或更高的最终精度。

