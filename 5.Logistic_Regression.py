import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)

    # losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
    # cost = losses.mean()
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Cost: {:.6f}".format(epoch, nb_epochs, cost.item()))

print(hypothesis[:5])    
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction[:5])
print(y_train[:5])

correct_prediction = prediction.float() == y_train
print(correct_prediction[:5])