import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


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


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        num_classes: int = 10,
        base_channels: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = base_channels

        # CIFAR-10 images are 32x32, so this variant uses a 3x3 stem and no max-pool.
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def resnet18_cifar(num_classes: int = 10, dropout: float = 0.0) -> ResNetCIFAR:
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dropout=dropout)


def build_transforms(use_randaugment: bool, erase_prob: float):
    train_ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaugment:
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    if erase_prob > 0:
        train_ops.append(transforms.RandomErasing(p=erase_prob, scale=(0.02, 0.15), ratio=(0.3, 3.3)))

    train_transform = transforms.Compose(train_ops)
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, test_transform


def build_loaders(args: argparse.Namespace):
    train_transform, test_transform = build_transforms(args.randaugment, args.random_erasing)
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
    return train_loader, test_loader


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
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
    grad_clip: float,
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
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    args: argparse.Namespace,
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
    }
    torch.save(checkpoint, output_dir / "last.pt")
    if is_best:
        torch.save(checkpoint, output_dir / "best.pt")


def append_log(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def make_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace):
    if args.warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    warmup = LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,
        total_iters=args.warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.min_lr,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 ResNet-18 variant from scratch.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/datasets/cifar-10-batches-py",
        help="CIFAR-10 directory. You may pass /datasets or /datasets/cifar-10-batches-py.",
    )
    parser.add_argument("--output-dir", type=str, default="./runs/resnet18_cifar10", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--randaugment", action="store_true", help="Enable RandAugment.")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--target-acc", type=float, default=0.0, help="Optional early-stop accuracy threshold. Disabled by default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if local files are missing.")
    parser.add_argument("--no-download", action="store_true", help="Do not download CIFAR-10.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.data_dir = str(resolve_cifar10_root(args.data_dir))

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_loader, test_loader = build_loaders(args)
    model = resnet18_cifar(dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = make_scheduler(optimizer, args)
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

    print(f"Device: {device}")
    print(f"CIFAR-10 root: {args.data_dir}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Training samples: {len(train_loader.dataset)}, test samples: {len(test_loader.dataset)}")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            args.grad_clip,
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "lr": f"{lr:.8f}",
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.3f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.3f}",
            "best_acc": f"{best_acc:.3f}",
            "time_sec": f"{elapsed:.1f}",
        }
        append_log(output_dir / "metrics.csv", row)
        save_checkpoint(output_dir, model, optimizer, scheduler, epoch, best_acc, args, is_best)

        marker = "*" if is_best else " "
        print(
            f"{marker} Epoch {epoch:03d}/{args.epochs} "
            f"lr={lr:.5f} train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% best={best_acc:.2f}% "
            f"time={elapsed:.1f}s"
        )

        if args.target_acc > 0 and best_acc >= args.target_acc:
            print(f"Target reached: best validation accuracy is {best_acc:.2f}% >= {args.target_acc:.2f}%.")
            break

    print(f"Finished. Best validation accuracy: {best_acc:.2f}%")
    if best_acc >= 85.0:
        print("Target reached: validation accuracy is above 85%.")
    else:
        print("Target not reached yet. Try more epochs, RandAugment, or a larger learning-rate schedule.")


if __name__ == "__main__":
    main()
