import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_transforms(image_size: int, use_autoaugment: bool):
    train_ops = [
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if use_autoaugment:
        train_ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_transform = transforms.Compose(train_ops)
    test_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, test_transform


def build_loaders(args: argparse.Namespace):
    train_transform, test_transform = build_transforms(args.image_size, args.autoaugment)
    data_dir = resolve_cifar10_root(args.data_dir)
    download = args.download and not args.no_download

    train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    return train_loader, test_loader, data_dir


def load_state_dict_from_file(model: nn.Module, weights_path: str) -> str:
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"Warning: ignored unexpected keys: {unexpected[:5]}")
    if missing:
        print(f"Warning: missing keys after loading weights: {missing[:5]}")
    return str(Path(weights_path).resolve())


def build_pretrained_resnet18(args: argparse.Namespace) -> tuple[nn.Module, str]:
    weights_source = "random_init"
    if args.weights_path:
        model = models.resnet18(weights=None)
        weights_source = load_state_dict_from_file(model, args.weights_path)
    elif args.pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
            weights_source = str(weights)
        except AttributeError:
            model = models.resnet18(pretrained=True)
            weights_source = "torchvision_pretrained=True"
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    if args.dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(in_features, 10))
    else:
        model.fc = nn.Linear(in_features, 10)
    return model, weights_source


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("fc."):
            parameter.requires_grad = trainable


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if name.startswith("fc."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    return optim.SGD(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr},
        ],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item() * 100.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits.detach(), targets), batch_size)

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits, targets), batch_size)

    return loss_meter.avg, acc_meter.avg


def append_log(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_acc: float,
    args: argparse.Namespace,
    weights_source: str,
    is_best: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
        "weights_source": weights_source,
    }
    torch.save(checkpoint, output_dir / "last.pt")
    if is_best:
        torch.save(checkpoint, output_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ImageNet pretrained ResNet-18 on CIFAR-10.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/datasets/cifar-10-batches-py",
        help="CIFAR-10 directory. You may pass /datasets or /datasets/cifar-10-batches-py.",
    )
    parser.add_argument("--output-dir", type=str, default="./runs/resnet18_finetune_cifar10")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the new classification head.")
    parser.add_argument("--backbone-lr", type=float, default=0.001, help="Learning rate for pretrained layers.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--autoaugment", action="store_true")
    parser.add_argument("--target-acc", type=float, default=0.0, help="Optional early-stop accuracy threshold. Disabled by default.")
    parser.add_argument("--weights-path", type=str, default="", help="Local official ImageNet ResNet-18 weights path.")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Disable ImageNet pretrained weights.")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if local files are missing.")
    parser.add_argument("--no-download", action="store_true", help="Do not download CIFAR-10.")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, data_root = build_loaders(args)
    args.data_dir = str(data_root)

    model, weights_source = build_pretrained_resnet18(args)
    set_backbone_trainable(model, args.freeze_backbone_epochs <= 0)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = make_optimizer(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("best_acc", 0.0))
        weights_source = checkpoint.get("weights_source", weights_source)

    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"CIFAR-10 root: {args.data_dir}")
    print(f"Weights source: {weights_source}")
    print(f"Image size: {args.image_size}")
    print(f"Training samples: {len(train_loader.dataset)}, test samples: {len(test_loader.dataset)}")

    for epoch in range(start_epoch, args.epochs + 1):
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
            print("Backbone unfrozen; fine-tuning all ResNet-18 layers.")

        start = time.time()
        stage = "head" if epoch <= args.freeze_backbone_epochs else "all"
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        lrs = [group["lr"] for group in optimizer.param_groups]
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "stage": stage,
            "backbone_lr": f"{lrs[0]:.8f}",
            "head_lr": f"{lrs[1]:.8f}",
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.3f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.3f}",
            "best_acc": f"{best_acc:.3f}",
            "time_sec": f"{elapsed:.1f}",
        }
        append_log(output_dir / "metrics.csv", row)
        save_checkpoint(output_dir, model, optimizer, scheduler, epoch, best_acc, args, weights_source, is_best)

        marker = "*" if is_best else " "
        print(
            f"{marker} Epoch {epoch:03d}/{args.epochs} stage={stage} "
            f"backbone_lr={lrs[0]:.6f} head_lr={lrs[1]:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% best={best_acc:.2f}% "
            f"time={elapsed:.1f}s"
        )

        if args.target_acc > 0 and best_acc >= args.target_acc:
            print(f"Target reached: best validation accuracy is {best_acc:.2f}% >= {args.target_acc:.2f}%.")
            break

    print(f"Finished. Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
