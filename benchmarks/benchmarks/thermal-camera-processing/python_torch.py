#!/usr/bin/env python3
import torch

def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W = 16, 1024, 1024

    raw = torch.rand((B, H, W), device=device, dtype=torch.float32)

    dark = 0.02 + 0.01 * torch.rand((H, W), device=device, dtype=torch.float32)
    ffc = 0.98 + 0.04 * torch.rand((H, W), device=device, dtype=torch.float32)
    gain = 1.50 + 0.50 * torch.rand((H, W), device=device, dtype=torch.float32)
    offset = -0.05 + 0.10 * torch.rand((H, W), device=device, dtype=torch.float32)

    lin = (raw - dark) * ffc
    radiance = lin * gain + offset
    radiance = torch.clamp(radiance, min=0.0)

    tempK = 273.15 + 80.0 * torch.log1p(radiance)

    mean_temp = float(tempK.mean().double())
    print(f"RESULT_ok MEAN_TEMP={mean_temp:.6f} device={'cuda' if device.type=='cuda' else 'cpu'}")


if __name__ == "__main__":
    main()