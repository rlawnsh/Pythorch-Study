import numpy as np
import torch

t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)

print('Rank of t:', t.ndim)
print('Shapte of t:', t.shape)

t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
print(t.dim())
print(t.shape)

# Vector + scalar
m1 = torch.FloatTensor([[1,2]])
m2 = torch.FloatTensor([3]) # 3 -> [[3, 3]]
print(m1 + m2)

# 1 x 2 Vector + 2 x 1 Vector 
m1 = torch.FloatTensor([[1,2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

# Multiplication vs Matrix Multiplication
m1 = torch.FloatTensor([[1,2], [3,4]])
m2 = torch.FloatTensor([[1], [2]])
print(m1.matmul(m2)) # 2 x 1
print(m1.mul(m2)) # 2 x 2

# Mean
t = torch.FloatTensor([[1,2], [3,4]])
print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))

# Max and Argmax
t = torch.FloatTensor([[1,2], [3,4]])
print(t.max())
print(t.max(dim=0))
print(t.max(dim=1))

# View(Reshape)
t = np.array([[[0,1,2],
                [3,4,5]],
                
                [[6,7,8],
                [9,10,11]]])
ft = torch.FloatTensor(t)

print(ft.view([-1, 3]))
print(ft.view([-1,1,3]))

# Squeeze(차원 날려주기)
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
print(ft.squeeze())
print(ft.squeeze().shape)

# Unsqueeze
ft = torch.FloatTensor([0, 1, 2])
print(ft)
print(ft.unsqueeze(0))
print(ft.view(1,-1))
print(ft.unsqueeze(1))
print(ft.unsqueeze(-1))

# Type Casting
lt = torch.LongTensor([1, 2, 3, 4])
print(lt.float())

# Concatenate
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5,6], [7, 8]])
print(torch.cat([x, y], dim=0))
print(torch.cat([x, y], dim=1))

# Stacking, Concatenate보다 간단
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))
print(torch.stack([x, y, z], dim=1))

print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))

# Ones and Zeros
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(torch.ones_like(x))
print(torch.zeros_like(x))

# In-place Operation
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.))
print(x)
print(x.mul_(2.))
print(x)

