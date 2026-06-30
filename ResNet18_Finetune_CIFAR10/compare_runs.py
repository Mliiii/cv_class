import argparse
import csv
import json
from pathlib import Path


def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def summarize(name: str, rows: list[dict], target_acc: float) -> dict:
    best_row = max(rows, key=lambda row: to_float(row, "val_acc"))
    best_acc = to_float(best_row, "val_acc")
    best_epoch = int(float(best_row["epoch"]))
    final_row = rows[-1]
    target_epoch = None
    for row in rows:
        if to_float(row, "val_acc") >= target_acc:
            target_epoch = int(float(row["epoch"]))
            break

    return {
        "name": name,
        "epochs": int(float(final_row["epoch"])),
        "final_val_acc": to_float(final_row, "val_acc"),
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "target_acc": target_acc,
        "target_epoch": target_epoch,
    }


def format_target_epoch(value) -> str:
    return str(value) if value is not None else "未达到"


def make_markdown(scratch: dict, finetune: dict) -> str:
    lines = [
        "# ResNet-18 CIFAR-10 对比结果",
        "",
        "| 训练方式 | 训练轮数 | 最终验证精度 | 最佳验证精度 | 最佳 epoch | 达到目标精度 epoch |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        (
            f"| {scratch['name']} | {scratch['epochs']} | {scratch['final_val_acc']:.3f}% | "
            f"{scratch['best_val_acc']:.3f}% | {scratch['best_epoch']} | {format_target_epoch(scratch['target_epoch'])} |"
        ),
        (
            f"| {finetune['name']} | {finetune['epochs']} | {finetune['final_val_acc']:.3f}% | "
            f"{finetune['best_val_acc']:.3f}% | {finetune['best_epoch']} | {format_target_epoch(finetune['target_epoch'])} |"
        ),
        "",
        "## 分析要点",
        "",
        "- 最终分类精度：比较两个实验的最佳验证精度，数值更高说明最终泛化能力更强。",
        "- 收敛速度：比较达到目标精度所需的 epoch，epoch 越小表示收敛越快。",
        "- 迁移学习通常会在前几个 epoch 获得较高精度，因为 ImageNet 预训练权重已经学习了通用边缘、纹理和形状特征。",
        "- 从零训练需要从随机初始化开始学习底层视觉特征，因此早期收敛速度通常更慢，但训练充分时也可以达到较高精度。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare scratch ResNet-18 and ImageNet fine-tuned ResNet-18.")
    parser.add_argument(
        "--scratch-metrics",
        type=str,
        default="/workspace/project/Computer_Vision/ResNet-18/runs/resnet18_cifar10/metrics.csv",
    )
    parser.add_argument(
        "--finetune-metrics",
        type=str,
        default="./runs/resnet18_finetune_cifar10/metrics.csv",
    )
    parser.add_argument("--target-acc", type=float, default=85.0)
    parser.add_argument("--output", type=str, default="./runs/comparison.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scratch = summarize("从零训练", read_metrics(Path(args.scratch_metrics)), args.target_acc)
    finetune = summarize("ImageNet 预训练微调", read_metrics(Path(args.finetune_metrics)), args.target_acc)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = make_markdown(scratch, finetune)
    output_path.write_text(markdown, encoding="utf-8")
    output_path.with_suffix(".json").write_text(
        json.dumps({"scratch": scratch, "finetune": finetune}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(markdown)
    print(f"Saved comparison to: {output_path}")


if __name__ == "__main__":
    main()
