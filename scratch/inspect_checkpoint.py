import torch
from pathlib import Path

checkpoint = torch.load(Path(__file__).resolve().parents[1] / 'models' / 'lstm_rul.pth', map_location='cpu', weights_only=True)
if isinstance(checkpoint, dict) and 'config' in checkpoint:
    print(checkpoint['config'])
else:
    print("No config in checkpoint")
