'''
Need to set the initial weight values wisely
- Not all 0's
- Challenging issue
- Hinton et al. (2006) "A Fast Learning Algorithm for Deep Belief Nets" - Restricted Boltzman Machine(RBM)

Xavier / He initialization

'''
# mnist_nn_xavier

import torch
from torch.nn.modules.activation import ReLU
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data
import random
import math

lr = 0.001
training_epochs = 15
batch_size = 100

# MNIST dataset
# root : 어느 경로에 MNIST 데이터가 있는지
# train : True - trainset을 불러올지 
#         False - testset을 불러올지
# transform : MNIST 이미지를 불러올 때 어떤 transform을 불러올지
#             pytorch : image는 0-1사이의 값을 갖고 Channel, Height, width순으로 값을 가짐
#             image : 0-255의 값으로 가지고 Height, width, channel 순서로 되어있음.
# toTensor는 image를 pytorch 값으로 변경해준다.
# download : MNIST가 root에 없으면 다운을 받겠다는 의미 
mnist_train = dsets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True)

# dataset loader
# dataset : 어떤 데이터를 로드할지
# batch_size : 몇개씩 잘라서 불러올지
# shuffle : 60000만장 중 100개씩 순서를 무작위로 가져올지에 대한 여부
# drop_last : batch size로 나누었을때 뒤에 개수가 안맞아떨어지는 경우 사용하지 않으면 True
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size = batch_size, shuffle=True, drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# MNIST data image of shape 28 * 28 = 784
linear1 = torch.nn.Linear(784, 256, bias=True).to(device)
linear2 = torch.nn.Linear(256, 256, bias=True).to(device)
linear3 = torch.nn.Linear(256, 10, bias=True).to(device)
relu = torch.nn.ReLU()

# xavier initialization
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)

# model
model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)
    for X, Y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch 
    
    print("Epoch: ", "%04d" % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    # 예측된 결과와 실제 test label 간의 맞는 정도
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())