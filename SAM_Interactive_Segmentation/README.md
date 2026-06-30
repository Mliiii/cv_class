# SAM 交互式分割

本文件夹对应“SAM 交互式分割”作业。代码支持两种主要提示方式：

- 点提示：使用正点和负点告诉模型“要分割哪里”和“不要分割哪里”。
- 框提示：使用边界框约束目标所在区域。

如果同时提供点和框，还可以运行 `point_box` 组合提示，观察约束更强时分割结果的变化。

## 目录结构

```text
SAM_Interactive_Segmentation/
├── README.md
├── requirements.txt
└── segment_with_sam.py
```

## 环境和权重

当前环境没有安装 `segment_anything`，因此需要先安装依赖并准备 SAM checkpoint：

```bash
pip install -r requirements.txt
```

常用模型类型：

| model type | checkpoint 示例 |
| --- | --- |
| `vit_b` | `sam_vit_b_01ec64.pth` |
| `vit_l` | `sam_vit_l_0b3195.pth` |
| `vit_h` | `sam_vit_h_4b8939.pth` |

建议课程作业用 `vit_b`，速度更快、显存压力更小。

## 快速检查

不加载 SAM，只检查图片和提示格式：

```bash
cd /workspace/project/Computer_Vision/SAM_Interactive_Segmentation
/workspace/miniconda3/envs/downtime66/bin/python segment_with_sam.py \
  --image /workspace/project/datasets/mnist_preview/train_000_label_5.png \
  --points 14,14,1 \
  --box 4,4,24,24 \
  --dry-run
```

运行后会生成：

```text
runs/sam_interactive/prompt_preview.png
```

## 点提示分割

```bash
/workspace/miniconda3/envs/downtime66/bin/python segment_with_sam.py \
  --image /path/to/image.jpg \
  --checkpoint /path/to/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --points 320,240,1
```

多点提示示例，其中 `1` 是正点，`0` 是负点：

```bash
/workspace/miniconda3/envs/downtime66/bin/python segment_with_sam.py \
  --image /path/to/image.jpg \
  --checkpoint /path/to/sam_vit_b_01ec64.pth \
  --points "320,240,1;420,240,0" \
  --mode point
```

## 框提示分割

```bash
/workspace/miniconda3/envs/downtime66/bin/python segment_with_sam.py \
  --image /path/to/image.jpg \
  --checkpoint /path/to/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --box 120,80,520,430 \
  --mode box
```

## 同时比较点提示和框提示

默认 `--mode all` 会分别运行点提示和框提示。如果同时提供点和框，还会额外运行组合提示：

```bash
/workspace/miniconda3/envs/downtime66/bin/python segment_with_sam.py \
  --image /path/to/image.jpg \
  --checkpoint /path/to/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --points "320,240,1;420,240,0" \
  --box 120,80,520,430 \
  --mode all
```

## 输出文件

默认输出目录：

```text
runs/sam_interactive
```

主要文件：

- `prompt_preview.png`：显示输入点和框的预览图。
- `point_overlay_*.png`：点提示分割结果。
- `box_overlay_*.png`：框提示分割结果。
- `point_box_overlay_*.png`：点和框组合提示结果。
- `*_mask_*.png`：二值 mask。
- `summary.json`：每个 mask 的分数和面积。
- `config.json`：运行参数。

## 报告分析建议

可以这样写对比：

- 点提示更灵活，用户只需点在目标上即可得到候选 mask，但当图像中有多个相似物体或目标边界复杂时，单个点可能产生歧义。
- 负点可以帮助排除不想要的区域，例如遮挡物、相邻物体或背景中相似纹理。
- 框提示的空间约束更强，通常能减少目标选择歧义，但框画得太松可能包含背景，框画得太紧可能截断目标。
- 点和框组合时，框限定搜索范围，点进一步说明目标位置，通常比单独点提示更稳定。
- 如果水果被遮挡，SAM 可能只分割可见区域；若要分出完整水果，需要额外先验或标注，不应只依赖单个点提示。

