import argparse
import json
import math
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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000) * torch.arange(half_dim, device=timesteps.device)
        exponent = exponent / max(half_dim - 1, 1)
        emb = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    def __init__(self, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(1, base_channels, 3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.down2_downsample = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.down3_downsample = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.down3 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up2_upsample = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up2 = ResidualBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up1_upsample = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, time_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(timesteps)
        x = self.init_conv(x)
        d1 = self.down1(x, t)
        d2 = self.down2_downsample(d1)
        d2 = self.down2(d2, t)
        d3 = self.down3_downsample(d2)
        d3 = self.down3(d3, t)

        mid = self.mid(d3, t)

        u2 = self.up2_upsample(mid)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2, t)
        u1 = self.up1_upsample(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1(u1, t)
        return self.out(u1)


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.size(0), *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps: int, device: torch.device):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]], dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = extract(self.sqrt_alpha_bars, timesteps, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alpha_bars, timesteps, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t_index: int) -> torch.Tensor:
        timesteps = torch.full((x.size(0),), t_index, device=x.device, dtype=torch.long)
        beta_t = extract(self.betas, timesteps, x.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alpha_bars, timesteps, x.shape)
        sqrt_recip_alpha = extract(self.sqrt_recip_alphas, timesteps, x.shape)
        predicted_noise = model(x, timesteps)
        model_mean = sqrt_recip_alpha * (x - beta_t * predicted_noise / sqrt_one_minus)

        if t_index == 0:
            return model_mean

        posterior_var = extract(self.posterior_variance, timesteps, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, batch_size: int, image_size: int = 28) -> torch.Tensor:
        model.eval()
        x = torch.randn(batch_size, 1, image_size, image_size, device=self.device)
        for t_index in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, t_index)
        return x.clamp(-1.0, 1.0)


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
    model: SimpleUNet,
    diffusion: GaussianDiffusion,
    feature_net: MNISTFeatureNet,
    device: torch.device,
    batch_size: int,
    max_items: int,
) -> torch.Tensor:
    features = []
    generated = 0
    while generated < max_items:
        current = min(batch_size, max_items - generated)
        fake = diffusion.sample(model, current)
        fake = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
        feats = feature_net.forward_features(fake.to(device)).cpu()
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


@torch.no_grad()
def save_samples(
    model: SimpleUNet,
    diffusion: GaussianDiffusion,
    output_path: Path,
    count: int = 64,
    nrow: int = 8,
) -> None:
    samples = diffusion.sample(model, count)
    samples = ((samples + 1.0) / 2.0).cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(samples, output_path, nrow=nrow, padding=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple DDPM on MNIST and compute FID.")
    parser.add_argument("--data-root", type=str, default="/workspace/project/datasets/data/MNIST")
    parser.add_argument("--output-dir", type=str, default="./runs/ddpm_mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
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

    model = SimpleUNet(base_channels=args.base_channels).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Using device: {device}")
    print(f"MNIST root for torchvision: {data_root}")
    print(f"Training samples: {len(train_set)}")
    print(f"Diffusion timesteps: {args.timesteps}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        loss_sum = 0.0
        seen = 0
        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)
            timesteps = torch.randint(0, args.timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(images)
            noisy_images = diffusion.q_sample(images, timesteps, noise)
            predicted_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * batch_size
            seen += batch_size

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_samples(model, diffusion, output_dir / "samples" / f"epoch_{epoch:03d}.png")

        torch.save(model.state_dict(), output_dir / "ddpm_last.pt")
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss={loss_sum / seen:.4f} time={time.time() - start:.1f}s"
        )

    save_samples(model, diffusion, output_dir / "samples" / "final.png")

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
            model,
            diffusion,
            feature_net,
            device,
            args.batch_size,
            args.fid_samples,
        )
        fid = frechet_distance(real_features, fake_features)
        result = {"fid": fid, "fid_samples": args.fid_samples, "feature": "MNISTFeatureNet-128"}
        (output_dir / "fid_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"FID: {fid:.4f}")


if __name__ == "__main__":
    main()
