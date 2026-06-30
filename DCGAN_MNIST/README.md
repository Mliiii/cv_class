# DCGAN on MNIST

本文件夹对应生成模型作业的方案 1：基于 PyTorch 在 MNIST 数据集上训练一个 DCGAN，并计算生成图像的 FID 分数。

## 数据路径

你当前的数据在：

```text
/workspace/project/datasets/data/MNIST
```

脚本默认使用这个路径。由于 `torchvision.datasets.MNIST` 需要传入 `MNIST` 文件夹的父目录，代码中已经自动兼容：

- 传 `/workspace/project/datasets/data/MNIST` 可以运行；
- 传 `/workspace/project/datasets/data` 也可以运行。

## 运行训练

```bash
cd /workspace/project/Computer_Vision/DCGAN_MNIST
/workspace/miniconda3/envs/downtime66/bin/python train.py
```

更明确的运行方式：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train.py \
  --data-root /workspace/project/datasets/data/MNIST \
  --epochs 20 \
  --batch-size 128 \
  --fid-samples 1000
```

如果只想快速检查代码是否能跑，可以先跳过 FID：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train.py --epochs 1 --skip-fid
```

## 输出文件

默认输出目录：

```text
runs/dcgan_mnist
```

主要文件：

- `generator_last.pt`：生成器权重。
- `discriminator_last.pt`：判别器权重。
- `samples/epoch_xxx.png`：每若干轮保存的生成样本。
- `samples/final.png`：最终生成样本。
- `fid_result.json`：FID 计算结果。
- `config.json`：训练参数。

## 模型说明

DCGAN 由生成器和判别器组成。生成器输入随机噪声向量，通过反卷积逐步上采样生成 `1x28x28` 的手写数字图像；判别器输入真实图像或生成图像，判断其是否来自真实数据分布。训练时二者交替优化，生成器逐渐学会生成更接近真实 MNIST 的图像。

## FID 说明

FID 的核心思想是比较真实图像和生成图像在特征空间中的分布差异。标准 FID 通常使用 ImageNet 预训练 Inception 网络提取特征，但本作业在离线 MNIST 环境下运行，因此代码会先训练一个轻量级 MNIST 分类特征网络 `MNISTFeatureNet`，再在其 128 维特征上计算 Fréchet 距离。

FID 越低，表示生成图像分布越接近真实图像分布。为了结果更稳定，可以把 `--fid-samples` 调大，例如：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train.py --epochs 50 --fid-samples 5000
```

