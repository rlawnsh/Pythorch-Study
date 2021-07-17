'''
Convolution - 이미지 위에서 stride 값 만큼 filter(kernel)을 이동시키면서
겹쳐지는 부분의 각 원소의 값을 곱해서 모두 더한 값을 출력으로 하는 연산

stride : filter를 한번에 얼마나 이동 할 것인가

입력 채널 1 / 출력 채널 1 / 커널 크기 3*3
conv = nn.Conv2d(1,1,3)

out = conv(inputs)
input type: torch.Tensor
input shape: (N * C * H * W) / (batch_size, channel, height, width)

'''
import torch
import torch.nn as nn

conv = nn.Conv2d(1, 1, 11, stride=4, padding=0)
inputs = torch.Tensor(1,1,227,227)
out = conv(inputs)
print(out.shape)

inputs = torch.Tensor(1, 1, 28, 28)
conv1 = nn.Conv2d(1, 5, 5)
pool = nn.MaxPool2d(2)
out = conv1(inputs)
out2 = pool(out)
print(out.size())
print(out2.size())