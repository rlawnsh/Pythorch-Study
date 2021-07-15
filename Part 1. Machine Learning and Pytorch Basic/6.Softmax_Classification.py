import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For reproducibility, 똑같은 결과 보장
torch.manual_seed(1)

z = torch.FloatTensor([1, 2, 3])

hypothesis = F.softmax(z, dim=0)
print(hypothesis)

# Softmax값의 합은 1이된다.
print(hypothesis.sum())

# Cross Entropy Loss
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)

y = torch.randint(5, (3,)).long()
print(y)

y_one_hot = torch.zeros_like(hypothesis)
print(y_one_hot.scatter_(1,y.unsqueeze(1),1))

# Low level
# cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
# High level, NLL = Negative Log Likelihood
# cost = F.nll_loss(F.log_softmax(z, dim=1), y)
# Pytorch also has F.cross_entropy that combines F.log_softmax() and F.nll_loss()
cost = F.cross_entropy(z, y)
print(cost)

