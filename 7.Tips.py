'''
Maximum Likelihood Estimation(MLE)

Optimization via Gradient Descent or Grdient Ascent

Overfitting 방지
- More Data
- Less features
- Regularization
    - Early Stopping
    - Reducingn Network Size
    - Weight Decay
    - Dropout
    - Batch Normalization

Basic Approach to Train DNN
1. Make a neural network architecture
2. Train and check that model is over-fitted
    a. If it is not, increase the model size(deeper and wider)
    b. If it is, add regularization, such as drop-out, batch-normalization.
3. Repeat from step-2

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1, 2, 1],
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7]])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])                             

x_test = torch.FloatTensor([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        
    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

optimizer = optim.SGD(model.parameters(), lr=0.1)

def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):
        # H(x) 계산
        prediction = model(x_train) 

        # cost 계산
        cost = F.cross_entropy(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)

    print('Accuracy: {}% Cost:  {:.6f}'.format(correct_count / len(y_test) * 100, cost.item()))

train(model, optimizer, x_train, y_train)

test(model, optimizer, x_test, y_test)
