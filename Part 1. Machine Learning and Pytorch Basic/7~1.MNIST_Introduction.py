'''
The torchvision package consists of popular datasets, model architectures,
 and common image transformations for computer vision.

In the neural network terminology:
- one epoch = one forward pass and one backward pass of all the training examples
- batch size = the number of training examples in one forward/backward pass. The
higher the batch size, the more memory space you'll need.
- number of iterations = number of passes, each pass using [batch size] number of
examples. To be clear, one pass = one forward pass + one backward pass (we do not
count the forward pass and backward pass as two different passes).

Example: if you have 1000 training examples, and your batch size is 500, then it will take 2
iterations to complete 1 epoch.

'''

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data
import random


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

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

linear = torch.nn.Linear(784, 10, bias=True).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)
    for X, Y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch 
    
    print("Epoch: ", "%04d" % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    # 예측된 결과와 실제 test label 간의 맞는 정도
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    
    # image로 보여주려함
    # Get one and predict , 무작위로 image하나를 뽑아서 예측을 해보기 위함
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)
    
    # 실제값
    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    # 예측값
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
