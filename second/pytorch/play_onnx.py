# Super Resolution model definition in PyTorch
import torch.nn as nn
import torch.nn.init as init
import torch

from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=64,
                               kernel_size=1,
                               stride=1)
        self.bn = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool1d(1, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.dense1 = nn.Linear(in_features=320, out_features=50)
        self.dense1_bn = nn.BatchNorm1d(50)
        self.dense2 = nn.Linear(50, 10)

        self.t_conv = nn.ConvTranspose2d(100, 1, (1,8), stride = (1, 7))
        self.conv3 = nn.Conv2d(9, 64, kernel_size=(1,10), stride = (1, 1), dilation = (1, 11))

    def forward(self, x):
        # print(x.size())
        # x = self.conv3(x)
        a = torch.ones(1, 9, 8599, 100)*0.5
        # mask = x > a
        # x = x*mask.float()
        x = x*a.float()
        # x = x.select(3, 0)
        print (x)
        print(x.size())

        return x
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        return x
        # x = x.view(-1, 320) #reshape
        # x = F.relu(self.dense1_bn(self.dense1(x)))
        # x = F.relu(self.dense2(x))
        # return F.log_softmax(x)


# A = torch.randn(1, 8995, 100, 9)
A = torch.randn(1, 100, 8599, 9)
A = torch.randn(1, 9, 8599, 100)
input = A
net = Net()
a = net(input)
# print(A)

torch_out = torch.onnx._export(net,             # model being run
                               input,                       # model input (or a tuple for multiple inputs)
                               "super_resolution.onnx")      # store the trained parameter weights inside the model file

print("finished")
