import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - w) // 2
    top = (size - h) // 2
    canvas.paste(img, (left, top))
    return canvas


class PosterDataset(Dataset):
    def __init__(self, root: Path, image_size: int):
        self.paths: List[Path] = sorted(
            [p for p in root.glob("*.jpg")] + [p for p in root.glob("*.png")]
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")

        self.transform = transforms.Compose(
            [
                transforms.Lambda(pad_to_square),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fall back to a different index if a file is corrupt.
            alt = (idx + 1) % len(self.paths)
            img = Image.open(self.paths[alt]).convert("RGB")
        return self.transform(img)


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def build_generator(nz: int, ngf: int, nc: int, image_size: int) -> nn.Sequential:
    n = int(math.log2(image_size))
    if 2 ** n != image_size or image_size < 32:
        raise ValueError("image_size must be a power of 2 and >= 32 (e.g., 64, 128).")

    layers: List[nn.Module] = []
    curr = ngf * (2 ** (n - 3))
    layers += [
        nn.ConvTranspose2d(nz, curr, 4, 1, 0, bias=False),
        nn.BatchNorm2d(curr),
        nn.ReLU(True),
    ]

    while curr > ngf:
        out = curr // 2
        layers += [
            nn.ConvTranspose2d(curr, out, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(True),
        ]
        curr = out

    layers += [
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        nn.Tanh(),
    ]
    return nn.Sequential(*layers)


def build_discriminator(ndf: int, nc: int, image_size: int) -> nn.Sequential:
    n = int(math.log2(image_size))
    if 2 ** n != image_size or image_size < 32:
        raise ValueError("image_size must be a power of 2 and >= 32 (e.g., 64, 128).")

    layers: List[nn.Module] = [
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    curr = ndf
    target = ndf * (2 ** (n - 3))

    while curr < target:
        out = curr * 2
        layers += [
            nn.Conv2d(curr, out, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        curr = out

    layers += [
        nn.Conv2d(curr, 1, 4, 1, 0, bias=False),
        nn.Sigmoid(),
    ]
    return nn.Sequential(*layers)


class PixelNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int, w_dim: int, n_layers: int = 8) -> None:
        super().__init__()
        layers = [PixelNorm()]
        for _ in range(n_layers):
            layers += [nn.Linear(w_dim, w_dim), nn.LeakyReLU(0.2, inplace=True)]
        self.layers = nn.Sequential(*layers)
        self.z_dim = z_dim
        self.w_dim = w_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[1] != self.w_dim:
            z = z.view(z.size(0), -1)
            if z.shape[1] != self.w_dim:
                raise ValueError("z dimension mismatch for mapping network.")
        return self.layers(z)


class NoiseInjection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
    ) -> None:
        super().__init__()
        self.eps = 1e-8
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.modulation = nn.Linear(style_dim, in_channels)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        style = self.modulation(style).view(batch, 1, self.in_channels, 1, 1)
        weight = self.weight * (style + 1.0)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        x = x.view(1, batch * self.in_channels, height, width)
        out = nn.functional.conv2d(x, weight, padding=self.kernel_size // 2, groups=batch)
        out = out.view(batch, self.out_channels, height, width)
        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size, style_dim)
        self.noise = NoiseInjection(out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, style)
        x = self.noise(x)
        return self.activate(x)


class ToRGB(nn.Module):
    def __init__(self, in_channels: int, style_dim: int) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, 3, 1, style_dim, demodulate=False)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return self.conv(x, style)


class StyleGenerator(nn.Module):
    def __init__(self, image_size: int, style_dim: int = 512, channel_multiplier: int = 2) -> None:
        super().__init__()
        if 2 ** int(math.log2(image_size)) != image_size or image_size < 32:
            raise ValueError("image_size must be a power of 2 and >= 32 (e.g., 64, 128).")

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 256 * channel_multiplier,
            64: 128 * channel_multiplier,
            128: 64 * channel_multiplier,
            256: 32 * channel_multiplier,
            512: 16 * channel_multiplier,
        }
        self.style_dim = style_dim
        self.image_size = image_size
        self.mapping = MappingNetwork(style_dim, style_dim)

        self.constant = nn.Parameter(torch.randn(1, channels[4], 4, 4))
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_ch = channels[4]
        self.convs.append(StyledConv(in_ch, in_ch, style_dim))
        self.to_rgbs.append(ToRGB(in_ch, style_dim))

        size = 8
        while size <= image_size:
            out_ch = channels.get(size, 16 * channel_multiplier)
            self.convs.append(StyledConv(in_ch, out_ch, style_dim))
            self.convs.append(StyledConv(out_ch, out_ch, style_dim))
            self.to_rgbs.append(ToRGB(out_ch, style_dim))
            in_ch = out_ch
            size *= 2

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        batch = z.size(0)
        x = self.constant.repeat(batch, 1, 1, 1)

        idx = 0
        x = self.convs[idx](x, w)
        rgb = self.to_rgbs[idx](x, w)
        idx += 1

        size = 8
        while size <= self.image_size:
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            x = self.convs[idx](x, w)
            idx += 1
            x = self.convs[idx](x, w)
            idx += 1
            rgb = nn.functional.interpolate(rgb, scale_factor=2, mode="nearest")
            rgb = rgb + self.to_rgbs[size.bit_length() - 3](x, w)
            size *= 2
        return torch.tanh(rgb)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activate(self.conv1(x))
        out = self.activate(self.conv2(out))
        out = nn.functional.avg_pool2d(out, 2)
        skip = nn.functional.avg_pool2d(self.skip(x), 2)
        return (out + skip) / math.sqrt(2.0)


class StyleDiscriminator(nn.Module):
    def __init__(self, image_size: int, channel_multiplier: int = 2) -> None:
        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 256 * channel_multiplier,
            64: 128 * channel_multiplier,
            128: 64 * channel_multiplier,
            256: 32 * channel_multiplier,
            512: 16 * channel_multiplier,
        }
        if image_size not in channels:
            raise ValueError("image_size not supported for StyleGAN discriminator.")

        in_ch = channels[image_size]
        self.from_rgb = nn.Conv2d(3, in_ch, 1)
        blocks = []
        size = image_size
        while size > 4:
            out_ch = channels[size // 2]
            blocks.append(DiscriminatorBlock(in_ch, out_ch))
            in_ch = out_ch
            size //= 2
        self.blocks = nn.Sequential(*blocks)
        self.final_conv = nn.Conv2d(in_ch, channels[4], 3, padding=1)
        self.final_dense = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(channels[4] * 4 * 4, channels[4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels[4], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.from_rgb(x)
        out = self.blocks(out)
        out = self.final_conv(out)
        return self.final_dense(out)


def save_grid(tensor: torch.Tensor, path: Path, nrow: int = 8) -> None:
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, value_range=(-1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(grid, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DCGAN on movie posters.")
    parser.add_argument("--data_dir", default="posters")
    parser.add_argument("--out_dir", default="gan_outputs")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--model", choices=["dcgan", "stylegan"], default="dcgan")
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--r1", type=float, default=1.0)
    parser.add_argument("--r1_every", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PosterDataset(data_dir, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.model == "stylegan":
        net_g = StyleGenerator(
            args.image_size, style_dim=args.style_dim, channel_multiplier=args.channel_multiplier
        ).to(device)
        net_d = StyleDiscriminator(
            args.image_size, channel_multiplier=args.channel_multiplier
        ).to(device)
        opt_d = optim.Adam(net_d.parameters(), lr=args.lr, betas=(0.0, 0.99))
        opt_g = optim.Adam(net_g.parameters(), lr=args.lr, betas=(0.0, 0.99))
        fixed_noise = torch.randn(64, args.style_dim, device=device)
    else:
        net_g = build_generator(args.nz, args.ngf, 3, args.image_size).to(device)
        net_d = build_discriminator(args.ndf, 3, args.image_size).to(device)
        net_g.apply(weights_init)
        net_d.apply(weights_init)
        opt_d = optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        opt_g = optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    config_path = out_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(vars(args), indent=2))

    for epoch in range(1, args.epochs + 1):
        for i, real in enumerate(loader, 1):
            real = real.to(device)
            bsz = real.size(0)
            if args.model == "stylegan":
                # Discriminator
                net_d.zero_grad(set_to_none=True)
                real_scores = net_d(real)
                loss_d_real = nn.functional.softplus(-real_scores).mean()

                noise = torch.randn(bsz, args.style_dim, device=device)
                fake = net_g(noise).detach()
                fake_scores = net_d(fake)
                loss_d_fake = nn.functional.softplus(fake_scores).mean()
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                opt_d.step()

                # R1 regularization
                if args.r1 > 0 and i % args.r1_every == 0:
                    real.requires_grad_(True)
                    real_scores = net_d(real)
                    r1_grads = torch.autograd.grad(
                        outputs=real_scores.sum(),
                        inputs=real,
                        create_graph=True,
                    )[0]
                    r1_penalty = r1_grads.pow(2).reshape(bsz, -1).sum(1).mean()
                    (args.r1 * 0.5 * r1_penalty).backward()
                    opt_d.step()

                # Generator
                net_g.zero_grad(set_to_none=True)
                noise = torch.randn(bsz, args.style_dim, device=device)
                fake = net_g(noise)
                fake_scores = net_d(fake)
                loss_g = nn.functional.softplus(-fake_scores).mean()
                loss_g.backward()
                opt_g.step()
            else:
                # DCGAN
                net_d.zero_grad(set_to_none=True)
                label = torch.full((bsz,), 1.0, device=device)
                output = net_d(real).view(-1)
                loss_d_real = nn.BCELoss()(output, label)
                loss_d_real.backward()

                noise = torch.randn(bsz, args.nz, 1, 1, device=device)
                fake = net_g(noise)
                label.fill_(0.0)
                output = net_d(fake.detach()).view(-1)
                loss_d_fake = nn.BCELoss()(output, label)
                loss_d_fake.backward()
                loss_d = loss_d_real + loss_d_fake
                opt_d.step()

                net_g.zero_grad(set_to_none=True)
                label.fill_(1.0)
                output = net_d(fake).view(-1)
                loss_g = nn.BCELoss()(output, label)
                loss_g.backward()
                opt_g.step()

            if i % 100 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | Step {i}/{len(loader)} "
                    f"| Loss_D {loss_d.item():.4f} | Loss_G {loss_g.item():.4f}"
                )

        with torch.no_grad():
            if args.model == "stylegan":
                fake = net_g(fixed_noise).detach().cpu()
            else:
                fake = net_g(fixed_noise).detach().cpu()
        save_grid(fake, out_dir / f"fake_epoch_{epoch:03d}.png")

        torch.save(
            {
                "epoch": epoch,
                "net_g": net_g.state_dict(),
                "net_d": net_d.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "args": vars(args),
            },
            out_dir / f"checkpoint_{epoch:03d}.pt",
        )


if __name__ == "__main__":
    main()
