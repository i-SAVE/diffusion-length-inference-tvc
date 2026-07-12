#!/usr/bin/env python3
"""Record the exact environment (versions, GPU, CUDA) into results/logs/environment.json.

Run this once on the machine used for the runs; the resulting file is part of the
archival package and lets a reviewer reproduce the numerical regime exactly.
"""
import json, os, platform, subprocess, sys

def _v(mod):
    try:
        m = __import__(mod)
        return getattr(m, '__version__', 'unknown')
    except Exception:
        return None

env = {
    'python': sys.version.split()[0],
    'platform': platform.platform(),
    'packages': {m: _v(m) for m in
                 ['torch', 'torchvision', 'numpy', 'scipy', 'matplotlib', 'umap', 'sklearn']},
}
try:
    import torch
    env['torch_cuda'] = {
        'available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count(),
    }
except Exception as e:
    env['torch_cuda'] = f'unavailable: {e}'
try:
    env['nvidia_smi'] = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
         '--format=csv,noheader'], text=True).strip()
except Exception:
    env['nvidia_smi'] = None

os.makedirs('results/logs', exist_ok=True)
out = 'results/logs/environment.json'
json.dump(env, open(out, 'w'), indent=2)
print(json.dumps(env, indent=2))
print('\nWrote', out)
