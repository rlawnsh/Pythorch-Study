import torch
from torch import optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# Weight와 Bias 0으로 초기화
# requires_grad=True : 학습할 것이라고 명시
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# [W, b]는 학습할 tensor들 / lr=0.01은 learning rate
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 2000
for epoch in range(1, nb_epochs + 1):

    hypothesis = x_train * W + b 

    # MSE(Mean Squard Error)구하기, torch.mean으로 평균 계산
    # 예측값과 실제값의 차이를 제곱한 값의 평균
    cost = torch.mean((hypothesis - y_train)**2)

    # zero_grad()로 gradient 초기화
    # backward()로 gradient 계산
    # step()으로 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(epoch, nb_epochs, W.item(), b.item(), cost.item()))
