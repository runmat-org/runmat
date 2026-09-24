"""Matched-precision PyTorch CPU/MPS pricing with explicit synchronization."""
import math
import os
import time

import numpy as np
import torch


def main():
    torch.set_num_threads(1)
    gpu = os.environ.get('MC_DEVICE') == 'gpu'
    device = torch.device('mps' if gpu else 'cpu')
    if gpu and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; refusing CPU fallback')
    dtype = torch.float32 if os.environ.get('MC_PRECISION') == 'float32' else torch.float64
    if os.environ.get('MC_MODE') != 'vectorized':
        raise ValueError('PyTorch variant supports vectorized mode only')
    z_host = torch.from_numpy(np.fromfile(os.environ['MC_FIXTURE'], dtype='<f8')).to(dtype=dtype)
    roundtrip = os.environ.get('MC_TIMING') == 'end-to-end'
    sync = torch.mps.synchronize if gpu else lambda: None
    z = z_host.to(device) if not roundtrip else z_host
    sync()
    print('DEVICE gpu MPS' if gpu else 'DEVICE cpu')
    repeats = int(os.environ.get('MC_REPEATS', '10'))
    warmups = int(os.environ.get('MC_WARMUPS', '3'))
    for rep in range(1, 1 + repeats + warmups):
        sync()
        start = time.perf_counter()
        if roundtrip:
            z = z_host.to(device)
        payoff = math.exp(-0.03) * torch.clamp(100 * torch.exp(0.01 + 0.2*z) - 100, min=0)
        price = payoff.mean().item()
        stderr = (payoff.std(correction=1) / math.sqrt(len(z))).item()
        sync()
        elapsed = (time.perf_counter() - start) * 1000
        print(f'SAMPLE rep={rep} ms={elapsed:.12g} price={price:.17g} stderr={stderr:.17g}')


if __name__ == '__main__':
    main()
