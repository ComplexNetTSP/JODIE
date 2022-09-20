import torch

device = "cuda"

torch.rand(100000000).to(device)