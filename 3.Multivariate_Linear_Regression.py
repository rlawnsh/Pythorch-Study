import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# nn.Module을 상속해서 모델 생성
# nn.Linear(3, 1) 입력 차원: 3 / 출력 차원: 1
# Hypothesis 계산은 forward() 에서
# Gradient계산은 PyTorch가 알아서 해준다 backward()
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 초기화
model = MultivariateLinearRegressionModel()

optimizer = optim.SGD(model.parameters(), lr = 1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

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

    # detach(): 기존 Tensor에서 gradient 전파가 안되는 텐서 생성
    print('Epoch {:4d}/{} Prediction: {} Cost: {:.6f}'.format(epoch, nb_epochs, prediction.squeeze().detach(), cost.item()))
