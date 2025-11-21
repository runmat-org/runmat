#!/usr/bin/env python3
import os
import torch


def build_lcg_image(batch: int, height: int, width: int, seed: int = 0) -> torch.Tensor:
    """Deterministic uint32 LCG field that matches the RunMat parity input."""
    cpu = torch.device("cpu")
    bid = torch.arange(batch, dtype=torch.int64, device=cpu).view(batch, 1, 1)
    yid = torch.arange(height, dtype=torch.int64, device=cpu).view(1, height, 1)
    xid = torch.arange(width, dtype=torch.int64, device=cpu).view(1, 1, width)
    stride_hw = torch.tensor(height * width, dtype=torch.int64, device=cpu)
    stride_w = torch.tensor(width, dtype=torch.int64, device=cpu)
    seed32 = torch.tensor(int(seed), dtype=torch.int64, device=cpu)
    idx = bid * stride_hw + yid * stride_w + xid + seed32
    state = (torch.tensor(1664525, dtype=torch.int64, device=cpu) * idx +
             torch.tensor(1013904223, dtype=torch.int64, device=cpu)) % torch.tensor(4294967296, dtype=torch.int64, device=cpu)
    imgs = state.to(torch.float32) / torch.tensor(4294967296.0, dtype=torch.float32, device=cpu)
    return imgs


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    B, H, W = 16, 2160, 3840
    B = int(os.environ.get("IMG_B", B))
    H = int(os.environ.get("IMG_H", H))
    W = int(os.environ.get("IMG_W", W))
    gain = torch.tensor(1.0123, dtype=torch.float32)
    bias = torch.tensor(-0.02, dtype=torch.float32)
    gamma = torch.tensor(1.8, dtype=torch.float32)
    eps0 = torch.tensor(1e-6, dtype=torch.float32)

    imgs = build_lcg_image(B, H, W).to(torch.float32)
    mu = imgs.mean(dim=(1, 2), keepdim=True)
    sigma = ((imgs - mu).pow(2).mean(dim=(1, 2), keepdim=True) + eps0).sqrt()

    out = ((imgs - mu) / sigma) * gain + bias
    out = torch.maximum(out, torch.tensor(0.0, dtype=torch.float32))
    out = out.pow(gamma)
    mse = (out - imgs).pow(2).mean().float().item()
    print(f"RESULT_ok MSE={mse:.6e} device=cpu")


if __name__ == "__main__":
    main()