#!/usr/bin/env python3
import os
import torch

def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W = 16, 2160, 3840
    gain = torch.tensor(1.0123, dtype=torch.float32, device=device)
    bias = torch.tensor(-0.02, dtype=torch.float32, device=device)
    gamma = torch.tensor(1.8, dtype=torch.float32, device=device)
    eps0 = torch.tensor(1e-6, dtype=torch.float32, device=device)

    imgs = torch.rand((B, H, W), device=device, dtype=torch.float32)
    mu = imgs.mean(dim=(1, 2), keepdim=True)
    sigma = ((imgs - mu).pow(2).mean(dim=(1, 2), keepdim=True) + eps0).sqrt()

    out = ((imgs - mu) / sigma) * gain + bias
    out = out.pow(gamma)
    mse = (out - imgs).pow(2).mean().double().item()
    dev = "cuda" if device.type == "cuda" else "cpu"
    print(f"RESULT_ok MSE={mse:.6e} device={dev}")

if __name__ == "__main__":
    main()