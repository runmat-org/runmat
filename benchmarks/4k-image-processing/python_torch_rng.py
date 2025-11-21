#!/usr/bin/env python3
import os
import torch

def main() -> None:
    torch.manual_seed(0)
    override = os.environ.get("TORCH_DEVICE")
    if override:
        device = torch.device(override)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    B, H, W = 16, 2160, 3840
    B = int(os.environ.get("IMG_B", B))
    H = int(os.environ.get("IMG_H", H))
    W = int(os.environ.get("IMG_W", W))
    gain = torch.tensor(1.0123, dtype=torch.float32, device=device)
    bias = torch.tensor(-0.02, dtype=torch.float32, device=device)
    gamma = torch.tensor(1.8, dtype=torch.float32, device=device)
    eps0 = torch.tensor(1e-6, dtype=torch.float32, device=device)

    imgs = torch.rand((B, H, W), device=device, dtype=torch.float32)
    mu = imgs.mean(dim=(1, 2), keepdim=True)
    sigma = ((imgs - mu).pow(2).mean(dim=(1, 2), keepdim=True) + eps0).sqrt()

    out = ((imgs - mu) / sigma) * gain + bias
    # Clamp to avoid NaNs from fractional power on negatives
    out = torch.maximum(out, torch.tensor(0.0, device=device, dtype=out.dtype))
    out = out.pow(gamma)
    # MPS does not support float64; keep result in float32 for portability
    mse = (out - imgs).pow(2).mean().float().item()
    print(f"RESULT_ok MSE={mse:.6e} device={device.type}")

if __name__ == "__main__":
    main()