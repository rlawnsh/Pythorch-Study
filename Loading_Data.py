# Minibatch Gradient Descent
# 전체 데이터를 균일하게 나눠서 학습

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                      [93, 88, 93],
                      [89, 91, 90],
                      [96, 98, 100],
                      [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x,y

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)
    
dataset = CustomDataset() 

# batch_size = 2 -> 각 minibatch의 크기
# shuffle = True -> Epoch 마다 데이터 셋을 섞어서, 데이터가 학습되는 순서를 바꾼다
dataloader = DataLoader(
    dataset,
    batch_size = 2,
    shuffle = True,
)

# 모델 초기화
model = MultivariateLinearRegressionModel()

optimizer = optim.SGD(model.parameters(), lr = 1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        # torch.nn.functional 에서 제공하는 loss function 사용
        # 쉽게 다른 loss와 교체 가능(l1_loss, smooth_l1_loss 등...)
        cost = F.mse_loss(prediction, y_train)

        # zero_grad()로 gradient 초기화
        # backward()로 gradient 계산
        # step()으로 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()))