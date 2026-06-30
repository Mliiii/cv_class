# 计算机视觉课程课后作业

* 姓名：李佳豪
* 学号：2251221137

| 目录 | 作业内容 | 数据集/输入 | 主要输出 |
| --- | --- | --- | --- |
| `ResNet-18/` | 从零训练 CIFAR-10 版 ResNet-18 变体 | CIFAR-10 | `best.pt`、`last.pt`、`metrics.csv` |
| `ResNet18_Finetune_CIFAR10/` | 使用 ImageNet 预训练 ResNet-18 在 CIFAR-10 上微调，并与从零训练对比 | CIFAR-10 | 微调权重、训练日志、对比结果 |
| `DCGAN_MNIST/` | 在 MNIST 上训练 DCGAN 生成手写数字，并计算 FID | MNIST | 生成图片、生成器/判别器权重、FID |
| `DDPM_MNIST/` | 在 MNIST 上训练简化 DDPM 扩散模型，并计算 FID | MNIST | 采样图片、DDPM 权重、FID |
| `CLIP_ZeroShot_CIFAR10/` | 使用 CLIP 对 CIFAR-10 做零样本分类 | CIFAR-10 | 准确率、混淆矩阵、失败案例 |
| `SAM_Interactive_Segmentation/` | 使用 SAM 进行点提示、框提示和组合提示分割 | 自定义图片 | 分割 mask、叠加图、提示预览 |
| `ViT精读/` | Vision Transformer 论文精读资料 | 论文和汇报文件 | PDF、PPTX |
| `3D Gaussian Splatting 原始论文精读报告.pdf` | 3DGS论文 | 论文笔记 | 无 |
