# DDPM on MNIST

本文件夹对应生成模型作业的方案 2：基于 PyTorch 在 MNIST 数据集上训练一个简化 DDPM 扩散模型，用于生成手写数字，并计算生成图像的 FID 分数。

## 数据路径

你当前的数据在：

```text
/workspace/project/datasets/data/MNIST
```

脚本默认使用这个路径。代码会自动兼容 `torchvision.datasets.MNIST` 的目录要求，因此传入下面两个路径都可以：

```text
/workspace/project/datasets/data/MNIST
/workspace/project/datasets/data
```

## 运行训练

```bash
cd /workspace/project/Computer_Vision/DDPM_MNIST
/workspace/miniconda3/envs/downtime66/bin/python train.py
```

更明确的运行方式：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train.py \
  --data-root /workspace/project/datasets/data/MNIST \
  --epochs 10 \
  --batch-size 128 \
  --timesteps 200 \
  --fid-samples 1000
```

如果只想快速检查代码是否能跑，可以先跳过 FID：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train.py --epochs 1 --timesteps 50 --skip-fid
```

## 输出文件

默认输出目录：

```text
runs/ddpm_mnist
```

主要文件：

- `ddpm_last.pt`：DDPM 噪声预测网络权重。
- `samples/epoch_xxx.png`：每若干轮保存的采样结果。
- `samples/final.png`：最终生成样本。
- `fid_result.json`：FID 计算结果。
- `config.json`：训练参数。

## 模型说明

DDPM 的训练目标是学习从带噪图像中预测噪声。训练时先从真实 MNIST 图像出发，随机选择扩散步数并加入高斯噪声；网络输入带噪图像和时间步，输出预测噪声。生成时从纯高斯噪声开始，按照反向扩散过程逐步去噪，最终得到手写数字图像。

本实现使用一个轻量 U-Net：

- 时间步使用正弦位置编码；
- 主干包含下采样、瓶颈层和上采样；
- 使用 skip connection 保留空间细节；
- 损失函数为噪声预测的 MSE。

## FID 说明

FID 用于衡量真实图像分布和生成图像分布之间的距离，数值越低越好。标准 FID 常用 ImageNet 预训练 Inception 网络提取特征，但本作业聚焦 MNIST 且需要离线可运行，因此代码会先训练轻量级 `MNISTFeatureNet`，再在 128 维特征空间中计算 Fréchet 距离。

为了获得更稳定的结果，可以增加训练轮数和 FID 样本数：

```bash
/workspace/miniconda3/envs/downtime66/bin/python train.py \
  --epochs 50 \
  --timesteps 300 \
  --fid-samples 5000
```

