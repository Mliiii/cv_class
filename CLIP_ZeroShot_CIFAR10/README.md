# CLIP 零样本分类：CIFAR-10

本文件夹对应“CLIP 零样本分类”作业。这里采用 CIFAR-10 作为数据集，因为它刚好包含 10 个类别：airplane、automobile、bird、cat、deer、dog、frog、horse、ship、truck，符合题目“选取任意 10 类感兴趣图片”的要求。

## 目录结构

```text
CLIP_ZeroShot_CIFAR10/
├── README.md
├── requirements.txt
├── download_clip_checkpoint.py
└── zero_shot_clip_cifar10.py
```

## 数据集路径

默认使用已经解压好的 CIFAR-10：

```text
/datasets/cifar-10-batches-py
```

代码已自动兼容 `torchvision.datasets.CIFAR10` 的路径要求，所以传 `/datasets` 或 `/datasets/cifar-10-batches-py` 都可以。

## 环境说明

当前环境里可用的是 `open_clip`，因此代码采用 `open_clip_torch` 加载 CLIP。默认模型为：

```text
ViT-B-32 + openai pretrained weights
```

如果本机没有缓存 CLIP 权重，第一次运行可能需要联网下载。离线情况下可以先准备好 open_clip 权重文件，然后用 `--checkpoint` 指定本地路径。


## 先下载 CLIP 权重

正式运行需要本地 CLIP 权重。建议放在：

```text
/workspace/project/Computer_Vision/CLIP_ZeroShot_CIFAR10/checkpoints/ViT-B-32.pt
```

可以运行：

```bash
cd /workspace/project/Computer_Vision/CLIP_ZeroShot_CIFAR10
/workspace/miniconda3/envs/downtime66/bin/python download_clip_checkpoint.py
```

如果你已经有权重文件，也可以运行正式分类时用 `--checkpoint /path/to/ViT-B-32.pt` 指定。

## 快速检查

只检查数据集、类别和提示词，不加载模型：

```bash
cd /workspace/project/Computer_Vision/CLIP_ZeroShot_CIFAR10
/workspace/miniconda3/envs/downtime66/bin/python zero_shot_clip_cifar10.py --dry-run
```

## 运行零样本分类

默认每类抽取 100 张测试图，共 1000 张：

```bash
cd /workspace/project/Computer_Vision/CLIP_ZeroShot_CIFAR10
/workspace/miniconda3/envs/downtime66/bin/python zero_shot_clip_cifar10.py
```

使用完整 CIFAR-10 测试集：

```bash
/workspace/miniconda3/envs/downtime66/bin/python zero_shot_clip_cifar10.py --samples-per-class 0
```

指定本地 CLIP 权重：

```bash
/workspace/miniconda3/envs/downtime66/bin/python zero_shot_clip_cifar10.py \
  --checkpoint /path/to/open_clip_checkpoint.pt
```

## 输出文件

默认输出目录：

```text
runs/clip_zeroshot_cifar10
```

主要结果：

- `summary.json`：整体准确率、逐类准确率、高置信失败案例。
- `predictions.csv`：每张图的真实类别、预测类别、置信度和 top-3 结果。
- `confusion_matrix.png`：混淆矩阵。
- `failure_examples.png`：高置信错误样例图。
- `config.json`：实验参数和 prompt 模板。

## 报告分析建议

可以从以下角度写分析：

- 总体准确率：观察 CLIP 在没有 CIFAR-10 训练的情况下，能否直接识别 10 类图像。
- 逐类准确率：通常 truck、ship、airplane 等形状明确的类别更容易识别；cat、dog、deer、horse 等动物类更容易混淆。
- 失败案例：重点查看 `failure_examples.png` 和 `predictions.csv` 中的高置信错误样例，分析原因可能包括图像分辨率低、主体太小、背景干扰、类别语义接近、prompt 表达不够贴合 CIFAR-10。
- Prompt 影响：本代码使用多个 prompt 模板做平均，可以对比只用单个 prompt 时的准确率变化。

