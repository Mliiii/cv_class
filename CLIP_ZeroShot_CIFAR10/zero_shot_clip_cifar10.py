import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


CIFAR10_TEXT_NAMES = {
    "airplane": "airplane",
    "automobile": "car",
    "bird": "bird",
    "cat": "cat",
    "deer": "deer",
    "dog": "dog",
    "frog": "frog",
    "horse": "horse",
    "ship": "ship",
    "truck": "truck",
}

PROMPT_TEMPLATES = [
    "a photo of the {}.",
    "a small photo of the {}.",
    "a blurry photo of the {}.",
    "a cropped photo of the {}.",
    "a low resolution photo of the {}.",
    "a photo of one {}.",
    "a close-up photo of the {}.",
]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = SCRIPT_DIR / "checkpoints" / "ViT-B-32.pt"
OPENAI_VIT_B32_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
)


def resolve_cifar10_root(path: str) -> Path:
    """torchvision.datasets.CIFAR10 expects the parent of cifar-10-batches-py."""
    root = Path(path).expanduser().resolve()
    if root.name == "cifar-10-batches-py" and root.is_dir():
        return root.parent
    if (root / "cifar-10-batches-py").is_dir():
        return root
    return root


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_balanced_indices(targets: list[int], num_classes: int, samples_per_class: int, seed: int) -> list[int]:
    per_class = [[] for _ in range(num_classes)]
    for index, label in enumerate(targets):
        per_class[int(label)].append(index)

    rng = random.Random(seed)
    selected = []
    for label, indices in enumerate(per_class):
        indices = indices[:]
        rng.shuffle(indices)
        if samples_per_class > 0:
            indices = indices[:samples_per_class]
        selected.extend(indices)
    rng.shuffle(selected)
    return selected


class CIFAR10ForCLIP(Dataset):
    def __init__(self, dataset: datasets.CIFAR10, indices: list[int], preprocess):
        self.dataset = dataset
        self.indices = indices
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        source_index = self.indices[item]
        image, label = self.dataset[source_index]
        return self.preprocess(image), int(label), int(source_index)


def load_open_clip(args: argparse.Namespace, device: torch.device):
    import open_clip

    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else DEFAULT_CHECKPOINT_PATH
    if checkpoint.exists():
        pretrained = None
    elif args.allow_download:
        pretrained = args.pretrained
        checkpoint = None
    else:
        raise FileNotFoundError(
            "CLIP checkpoint not found. Put the official ViT-B-32.pt at "
            f"{DEFAULT_CHECKPOINT_PATH}, or pass --checkpoint /path/to/ViT-B-32.pt. "
            f"Download URL: {OPENAI_VIT_B32_URL}"
        )

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=pretrained,
        device=device,
    )
    if checkpoint is not None:
        open_clip.load_checkpoint(model, str(checkpoint), strict=True, device=device)
        print(f"Loaded CLIP checkpoint: {checkpoint}")

    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def build_text_features(model, tokenizer, class_names: list[str], device: torch.device) -> tuple[torch.Tensor, list[str]]:
    prompts = []
    for class_name in class_names:
        text_name = CIFAR10_TEXT_NAMES.get(class_name, class_name)
        prompts.extend(template.format(text_name) for template in PROMPT_TEMPLATES)

    text_tokens = tokenizer(prompts).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
    text_features = text_features.reshape(len(class_names), len(PROMPT_TEMPLATES), -1).mean(dim=1)
    text_features = F.normalize(text_features, dim=-1)
    return text_features, prompts


@torch.no_grad()
def run_inference(
    model,
    loader: DataLoader,
    text_features: torch.Tensor,
    class_names: list[str],
    device: torch.device,
) -> tuple[list[dict], np.ndarray]:
    rows = []
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    for images, labels, source_indices in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = 100.0 * image_features @ text_features.t()
        probs = logits.softmax(dim=-1)
        confidences, preds = probs.max(dim=1)
        top3 = probs.topk(k=min(3, len(class_names)), dim=1)

        for i in range(images.size(0)):
            true_label = int(labels[i].item())
            pred_label = int(preds[i].item())
            confusion[true_label, pred_label] += 1
            top3_labels = [class_names[int(idx)] for idx in top3.indices[i].cpu().tolist()]
            top3_scores = [float(score) for score in top3.values[i].cpu().tolist()]
            rows.append(
                {
                    "index": int(source_indices[i].item()),
                    "true_label": class_names[true_label],
                    "pred_label": class_names[pred_label],
                    "confidence": float(confidences[i].item()),
                    "correct": true_label == pred_label,
                    "top3": ";".join(f"{name}:{score:.4f}" for name, score in zip(top3_labels, top3_scores)),
                }
            )
    return rows, confusion


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "true_label", "pred_label", "confidence", "correct", "top3"])
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(path: Path, confusion: np.ndarray, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CLIP zero-shot confusion matrix on CIFAR-10")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_failure_grid(path: Path, dataset: datasets.CIFAR10, rows: list[dict], max_items: int) -> None:
    failures = [row for row in rows if not row["correct"]]
    failures.sort(key=lambda row: row["confidence"], reverse=True)
    failures = failures[:max_items]
    if not failures:
        return

    cell_w, cell_h = 176, 140
    cols = min(4, len(failures))
    rows_count = int(np.ceil(len(failures) / cols))
    canvas = Image.new("RGB", (cols * cell_w, rows_count * cell_h), "white")
    draw = ImageDraw.Draw(canvas)

    for k, row in enumerate(failures):
        col = k % cols
        row_id = k // cols
        x0 = col * cell_w
        y0 = row_id * cell_h
        image, _ = dataset[int(row["index"])]
        image = image.resize((96, 96), Image.Resampling.BICUBIC)
        canvas.paste(image, (x0 + 40, y0 + 4))
        draw.text((x0 + 6, y0 + 104), f"T: {row['true_label']}", fill=(0, 0, 0))
        draw.text((x0 + 6, y0 + 120), f"P: {row['pred_label']} {row['confidence']:.2f}", fill=(180, 0, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def summarize(rows: list[dict], confusion: np.ndarray, class_names: list[str]) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    per_class = {}
    for i, class_name in enumerate(class_names):
        class_total = int(confusion[i].sum())
        class_correct = int(confusion[i, i])
        per_class[class_name] = {
            "total": class_total,
            "correct": class_correct,
            "accuracy": class_correct / class_total if class_total else 0.0,
        }

    failures = [row for row in rows if not row["correct"]]
    failures.sort(key=lambda row: row["confidence"], reverse=True)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "per_class": per_class,
        "high_confidence_failures": failures[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP zero-shot classification on CIFAR-10.")
    parser.add_argument("--data-dir", type=str, default="/datasets/cifar-10-batches-py")
    parser.add_argument("--output-dir", type=str, default="./runs/clip_zeroshot_cifar10")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional local open_clip checkpoint.")
    parser.add_argument("--allow-download", action="store_true", help="Allow open_clip to download pretrained weights.")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--samples-per-class", type=int, default=100, help="Use <=0 for all samples.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--failure-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="Only check data and prompts; do not load CLIP.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_cifar10_root(args.data_dir)
    dataset = datasets.CIFAR10(root=str(data_root), train=args.split == "train", download=False)
    class_names = dataset.classes
    indices = select_balanced_indices(dataset.targets, len(class_names), args.samples_per_class, args.seed)

    print(f"CIFAR-10 root: {data_root}")
    print(f"Split: {args.split}")
    print(f"Classes: {class_names}")
    print(f"Selected images: {len(indices)}")

    config = vars(args).copy()
    config["resolved_data_root"] = str(data_root)
    config["classes"] = class_names
    config["prompt_templates"] = PROMPT_TEMPLATES
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.dry_run:
        for class_name in class_names:
            text_name = CIFAR10_TEXT_NAMES.get(class_name, class_name)
            print(f"{class_name}: {PROMPT_TEMPLATES[0].format(text_name)}")
        return

    device = torch.device(args.device)
    try:
        model, preprocess, tokenizer = load_open_clip(args, device)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Run: /workspace/miniconda3/envs/downtime66/bin/python download_clip_checkpoint.py")
        raise SystemExit(2)
    clip_dataset = CIFAR10ForCLIP(dataset, indices, preprocess)
    loader = DataLoader(
        clip_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=str(device).startswith("cuda"),
    )

    text_features, prompts = build_text_features(model, tokenizer, class_names, device)
    rows, confusion = run_inference(model, loader, text_features, class_names, device)
    summary = summarize(rows, confusion, class_names)

    save_csv(output_dir / "predictions.csv", rows)
    save_confusion_matrix(output_dir / "confusion_matrix.png", confusion, class_names)
    save_failure_grid(output_dir / "failure_examples.png", dataset, rows, args.failure_examples)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Accuracy: {summary['accuracy'] * 100:.2f}% ({summary['correct']}/{summary['total']})")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
