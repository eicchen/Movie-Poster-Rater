import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from train_gan import build_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate posters from a GAN checkpoint.")
    parser.add_argument("--checkpoint", default="gan_outputs/checkpoint_050.pt")
    parser.add_argument("--out_dir", default="gan_outputs/samples")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["args"]

    net_g = build_generator(cfg["nz"], cfg["ngf"], 3, cfg["image_size"]).to(device)
    net_g.load_state_dict(ckpt["net_g"])
    net_g.eval()

    noise = torch.randn(args.count, cfg["nz"], 1, 1, device=device)
    with torch.no_grad():
        fake = net_g(noise).cpu()

    out_path = out_dir / f"samples_{args.count}.png"
    nrow = min(5, args.count)
    save_image(fake, out_path, nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
