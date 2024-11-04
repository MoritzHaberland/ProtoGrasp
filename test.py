import torch
def hey():
    a = torch.randn(1, 2, 3, 4, 5)
    print(a)
    print(a.numel())

hey()