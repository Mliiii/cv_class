import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_mnist_root(path: str) -> Path:
    """torchvision.datasets.MNIST wants the parent directory of the MNIST folder."""
    root = Path(path).expanduser().resolve()
    if root.name == "MNIST" and (root / "raw").exists():
        return root.parent
    if (root / "MNIST" / "raw").exists():
        return root
    return root


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, feature_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.Conv2d(feature_maps, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, feature_maps: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(feature_maps * 4 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)


def init_dcgan_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class MNISTFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return F.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.forward_features(x))


def train_feature_extractor(
    data_root: Path,
    ckpt_path: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    epochs: int,
) -> MNISTFeatureNet:
    model = MNISTFeatureNet().to(device)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        return model

    train_set = datasets.MNIST(
        root=str(data_root),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        correct = 0
        total = 0
        loss_sum = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
        print(
            f"FID feature net epoch {epoch}/{epochs}: "
            f"loss={loss_sum / total:.4f}, acc={100.0 * correct / total:.2f}%"
        )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    model.eval()
    return model


@torch.no_grad()
def collect_real_features(
    feature_net: MNISTFeatureNet,
    data_root: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    max_items: int,
) -> torch.Tensor:
    test_set = datasets.MNIST(
        root=str(data_root),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    features = []
    seen = 0
    for images, _ in loader:
        images = images.to(device)
        feats = feature_net.forward_features(images).cpu()
        features.append(feats)
        seen += images.size(0)
        if seen >= max_items:
            break
    return torch.cat(features, dim=0)[:max_items]


@torch.no_grad()
def collect_fake_features(
    generator: Generator,
    feature_net: MNISTFeatureNet,
    device: torch.device,
    latent_dim: int,
    batch_size: int,
    max_items: int,
) -> torch.Tensor:
    features = []
    generated = 0
    while generated < max_items:
        current = min(batch_size, max_items - generated)
        z = torch.randn(current, latent_dim, 1, 1, device=device)
        fake = generator(z)
        fake = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
        feats = feature_net.forward_features(fake).cpu()
        features.append(feats)
        generated += current
    return torch.cat(features, dim=0)


def matrix_sqrt_psd(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    matrix = (matrix + matrix.t()) / 2.0
    eye = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    values, vectors = torch.linalg.eigh(matrix + eps * eye)
    values = torch.clamp(values, min=0.0)
    return (vectors * torch.sqrt(values).unsqueeze(0)) @ vectors.t()


def frechet_distance(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    real = real_features.double()
    fake = fake_features.double()
    mu_real = real.mean(dim=0)
    mu_fake = fake.mean(dim=0)
    real_centered = real - mu_real
    fake_centered = fake - mu_fake
    cov_real = real_centered.t().mm(real_centered) / (real.size(0) - 1)
    cov_fake = fake_centered.t().mm(fake_centered) / (fake.size(0) - 1)

    sqrt_cov_real = matrix_sqrt_psd(cov_real)
    cov_mean = matrix_sqrt_psd(sqrt_cov_real.mm(cov_fake).mm(sqrt_cov_real))
    fid = (mu_real - mu_fake).dot(mu_real - mu_fake)
    fid = fid + torch.trace(cov_real + cov_fake - 2.0 * cov_mean)
    return float(torch.clamp(fid, min=0.0).item())


def save_generator_samples(
    generator: Generator,
    noise: torch.Tensor,
    output_path: Path,
    nrow: int = 8,
) -> None:
    generator.eval()
    with torch.no_grad():
        samples = generator(noise).cpu()
        samples = (samples + 1.0) / 2.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(samples, output_path, nrow=nrow, padding=2)
    generator.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCGAN on MNIST and compute FID.")
    parser.add_argument("--data-root", type=str, default="/workspace/project/datasets/data/MNIST")
    parser.add_argument("--output-dir", type=str, default="./runs/dcgan_mnist")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--g-features", type=int, default=64)
    parser.add_argument("--d-features", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--fid-samples", type=int, default=1000)
    parser.add_argument("--fid-classifier-epochs", type=int, default=3)
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    data_root = resolve_mnist_root(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_set = datasets.MNIST(root=str(data_root), train=True, download=False, transform=transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    generator = Generator(args.latent_dim, args.g_features).to(device)
    discriminator = Discriminator(args.d_features).to(device)
    generator.apply(init_dcgan_weights)
    discriminator.apply(init_dcgan_weights)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    print(f"Using device: {device}")
    print(f"MNIST root for torchvision: {data_root}")
    print(f"Training samples: {len(train_set)}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        seen = 0
        for real_images, _ in train_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            discriminator.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images)
            d_real_loss = criterion(real_logits, real_labels)
            z = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_images = generator(z)
            fake_logits = discriminator(fake_images.detach())
            d_fake_loss = criterion(fake_logits, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_d.step()

            generator.zero_grad(set_to_none=True)
            fake_logits_for_g = discriminator(fake_images)
            g_loss = criterion(fake_logits_for_g, real_labels)
            g_loss.backward()
            optimizer_g.step()

            d_loss_sum += d_loss.item() * batch_size
            g_loss_sum += g_loss.item() * batch_size
            seen += batch_size

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_generator_samples(
                generator,
                fixed_noise,
                output_dir / "samples" / f"epoch_{epoch:03d}.png",
            )

        torch.save(generator.state_dict(), output_dir / "generator_last.pt")
        torch.save(discriminator.state_dict(), output_dir / "discriminator_last.pt")
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"d_loss={d_loss_sum / seen:.4f} g_loss={g_loss_sum / seen:.4f} "
            f"time={time.time() - start:.1f}s"
        )

    save_generator_samples(generator, fixed_noise, output_dir / "samples" / "final.png")

    if not args.skip_fid:
        feature_net = train_feature_extractor(
            data_root=data_root,
            ckpt_path=output_dir / "mnist_feature_net.pt",
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            epochs=args.fid_classifier_epochs,
        )
        real_features = collect_real_features(
            feature_net,
            data_root,
            device,
            args.batch_size,
            args.workers,
            args.fid_samples,
        )
        fake_features = collect_fake_features(
            generator,
            feature_net,
            device,
            args.latent_dim,
            args.batch_size,
            args.fid_samples,
        )
        fid = frechet_distance(real_features, fake_features)
        result = {"fid": fid, "fid_samples": args.fid_samples, "feature": "MNISTFeatureNet-128"}
        (output_dir / "fid_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"FID: {fid:.4f}")


if __name__ == "__main__":
    main()
