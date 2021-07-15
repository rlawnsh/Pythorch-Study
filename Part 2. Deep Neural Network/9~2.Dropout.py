'''
Overfitting
- Very high accuracy on the training dataset
- Poor accuracy on the test dataset

Dropout
- 사용할 노드와 사용하지 않을 노드를 무작위로 결정해서 전파가 이루어 진다.
- 여러 형태의 네트워크들이 최종 결과를 내는 앙상블의 효과가 있다.

'''
# mnist_nn_dropout

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
drop_prob = 0.3

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
linear1 = torch.nn.Linear(784, 512, bias=True).to(device)
linear2 = torch.nn.Linear(512, 512, bias=True).to(device)
linear3 = torch.nn.Linear(512, 512, bias=True).to(device)
linear4 = torch.nn.Linear(512, 512, bias=True).to(device)
linear5 = torch.nn.Linear(512, 10, bias=True).to(device)
relu = torch.nn.ReLU()
dropout= torch.nn.Dropout(p = drop_prob)

# xavier initialization
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)


# model
model = torch.nn.Sequential(linear1, relu, dropout, 
                            linear2, relu, dropout,
                            linear3, relu, dropout,
                            linear4, relu, dropout,
                            linear5).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)

model.train()   # set the model to train mode( dropout=True )
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
    model.eval()    # set the model to evaluation mode ( dropout=False )
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    # 예측된 결과와 실제 test label 간의 맞는 정도
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
