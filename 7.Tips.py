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
y_test = torch.FloatTensor([2, 2, 2])

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        
    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

optimizer = optim.SGD(model.parameters(), lr=0.1)