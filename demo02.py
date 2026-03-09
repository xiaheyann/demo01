import torch

a = torch.tensor([1, 2, 3],device="npu")
b = torch.tensor([4, 5, 6],device="npu")
c = a + b
print(c)