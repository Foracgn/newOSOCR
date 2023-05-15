import torch

x = torch.arange(2 * 3 * 4).view(1, 2, 3, 4)

print(x)
print(x.size()[1])
print(x.transpose(1, 0).shape)

y = x.view(1, 2, -1).sum(2).view(1, 2, 1, 1)

print(y)
print(x / y)

z = torch.arange(1, 25).view(1, 2, 3, 4)

A = x.view(1, 1, 2, 3, 4) * z.view(1, 2, 1, 3, 4)
print(A)
print(A.shape)
B = A.view(1, 2, 2, -1).sum(3).transpose(1, 0)
print(B)
print(B.shape)

linear = torch.nn.Linear(2, 512)
C = linear(B)

print(C)
print(C.shape)
