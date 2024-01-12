import torch
import torch.nn as nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
# import onnx
import numpy as np
# from onnx import helper, numpy_helper
from copy import deepcopy


class MNISTConv(nn.Module):
    def __init__(self):
        super(MNISTConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (4, 4), stride=2)
        self.conv2 = nn.Conv2d(16, 32, (4, 4), stride=2)
        self.linear1 = nn.Linear(800, 100)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        y1 = nn.ReLU()(self.conv1(x))
        y2 = nn.ReLU()(self.conv2(y1))
        y3 = torch.flatten(y2, start_dim=1)
        y4 = nn.ReLU()(self.linear1(y3))
        y5 = self.linear2(y4)
        return y5

    def forward_detail(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(nn.ReLU()(y1))
        y3 = torch.flatten(nn.ReLU()(y2), start_dim=1)
        y4 = self.linear1(y3)
        y5 = self.linear2(nn.ReLU()(y4))
        return [y1.clone().detach(), y2.clone().detach(), y4.clone().detach()]


class MNISTConvMed(nn.Module):
    def __init__(self):
        super(MNISTConvMed, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, (4, 4), stride=(2, 2), padding=(1, 1))
        self.linear1 = nn.Linear(1568, 1000)
        self.linear2 = nn.Linear(1000, 10)

    def forward(self, x):
        y1 = nn.ReLU()(self.conv1(x))
        y2 = nn.ReLU()(self.conv2(y1))
        y3 = torch.flatten(y2, start_dim=1)
        y4 = nn.ReLU()(self.linear1(y3))
        y5 = self.linear2(y4)
        return y5


class CIFARConv(nn.Module):
    def __init__(self):
        super(CIFARConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (4, 4), stride=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, (4, 4), stride=(2, 2))
        self.linear1 = nn.Linear(1152, 100)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        y1 = nn.ReLU()(self.conv1(x))
        y2 = nn.ReLU()(self.conv2(y1))
        y3 = torch.flatten(y2, start_dim=1)
        y4 = nn.ReLU()(self.linear1(y3))
        y5 = self.linear2(y4)
        return y5


class CIFARConvMed(nn.Module):
    def __init__(self):
        super(CIFARConvMed, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, (4, 4), stride=(2, 2), padding=(1, 1))
        self.linear1 = nn.Linear(2048, 1000)
        self.linear2 = nn.Linear(1000, 10)

    def forward(self, x):
        y1 = nn.ReLU()(self.conv1(x))
        y2 = nn.ReLU()(self.conv2(y1))
        y3 = torch.flatten(y2, start_dim=1)
        y4 = nn.ReLU()(self.linear1(y3))
        y5 = self.linear2(y4)
        return y5


class M6100(nn.Module):
    def __init__(self):
        super(M6100, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, 100)
        self.linear5 = nn.Linear(100, 100)
        self.linear6 = nn.Linear(100, 10)

    def forward(self, x):
        y0 = torch.flatten(x, start_dim=1)
        y1 = nn.ReLU()(self.linear1(y0))
        y2 = nn.ReLU()(self.linear2(y1))
        y3 = nn.ReLU()(self.linear3(y2))
        y4 = nn.ReLU()(self.linear4(y3))
        y5 = nn.ReLU()(self.linear5(y4))
        y6 = nn.ReLU()(self.linear6(y5))
        return y6


class M9200(nn.Module):
    def __init__(self):
        super(M9200, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 200)
        self.linear4 = nn.Linear(200, 200)
        self.linear5 = nn.Linear(200, 200)
        self.linear6 = nn.Linear(200, 200)
        self.linear7 = nn.Linear(200, 200)
        self.linear8 = nn.Linear(200, 200)
        self.linear9 = nn.Linear(200, 10)

    def forward(self, x):
        y0 = torch.flatten(x, start_dim=1)
        y1 = nn.ReLU()(self.linear1(y0))
        y2 = nn.ReLU()(self.linear2(y1))
        y3 = nn.ReLU()(self.linear3(y2))
        y4 = nn.ReLU()(self.linear4(y3))
        y5 = nn.ReLU()(self.linear5(y4))
        y6 = nn.ReLU()(self.linear6(y5))
        y7 = nn.ReLU()(self.linear7(y6))
        y8 = nn.ReLU()(self.linear8(y7))
        y9 = nn.ReLU()(self.linear9(y8))

        return y9


class C6100(nn.Module):
    def __init__(self):
        super(C6100, self).__init__()
        self.linear1 = nn.Linear(3 * 32 * 32, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, 100)
        self.linear5 = nn.Linear(100, 100)
        self.linear6 = nn.Linear(100, 100)
        self.linear7 = nn.Linear(100, 10)

    def forward(self, x):
        y0 = torch.flatten(x, start_dim=1)
        y1 = nn.ReLU()(self.linear1(y0))
        y2 = nn.ReLU()(self.linear2(y1))
        y3 = nn.ReLU()(self.linear3(y2))
        y4 = nn.ReLU()(self.linear4(y3))
        y5 = nn.ReLU()(self.linear5(y4))
        y6 = nn.ReLU()(self.linear6(y5))
        y7 = nn.ReLU()(self.linear7(y6))
        return y7


class C9200(nn.Module):
    def __init__(self):
        super(C9200, self).__init__()
        self.linear1 = nn.Linear(3 * 32 * 32, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 200)
        self.linear4 = nn.Linear(200, 200)
        self.linear5 = nn.Linear(200, 200)
        self.linear6 = nn.Linear(200, 200)
        self.linear7 = nn.Linear(200, 200)
        self.linear8 = nn.Linear(200, 200)
        self.linear9 = nn.Linear(200, 200)
        self.linear10 = nn.Linear(200, 10)

    def forward(self, x):
        y0 = torch.flatten(x, start_dim=1)
        y1 = nn.ReLU()(self.linear1(y0))
        y2 = nn.ReLU()(self.linear2(y1))
        y3 = nn.ReLU()(self.linear3(y2))
        y4 = nn.ReLU()(self.linear4(y3))
        y5 = nn.ReLU()(self.linear5(y4))
        y6 = nn.ReLU()(self.linear6(y5))
        y7 = nn.ReLU()(self.linear7(y6))
        y8 = nn.ReLU()(self.linear8(y7))
        y9 = nn.ReLU()(self.linear9(y8))
        y10 = nn.ReLU()(self.linear10(y9))

        return y10


if __name__ == '__main__':
    # onnx_model = onnx.load('./mnist_convSmallRELU__Point.onnx')
    onnx_model = onnx.load('./onnx_model/mnist_relu_6_100.onnx')
    # for e in onnx_model.graph.node:
    #     print(e.name, e.op_type)
    #     print(e)
    #     print('-'*40)

    raw_paras = []
    for e in onnx_model.graph.initializer:
        raw_paras.append(numpy_helper.to_array(e))
    raw_bias = [raw_paras[i] for i in range(0, len(raw_paras), 2)]
    raw_weights = [raw_paras[i] for i in range(1, len(raw_paras), 2)]
    # print(len(raw_weights))
    # print(len(raw_bias))

    c0 = numpy_helper.to_array(onnx_model.graph.node[0].attribute[0].t)[0, :, :, 0]
    c2 = numpy_helper.to_array(onnx_model.graph.node[2].attribute[0].t)[0, :, :, 0]
    # for i in range(len(raw_weights)):
    #     print(raw_weights[i].shape, raw_bias[i].shape)
    # print(c0, c2)

    # for e in onnx_model.graph.node:
    #     print(e.name, e.op_type)

    test_set = MNIST('../data/', train=False, download=False, transform=ToTensor())
    # test_set = CIFAR10('../data/', train=False, download=False, transform=ToTensor())
    images = test_set.data / 255
    labels = test_set.targets

    # t_model = MNISTConv()
    t_model = M6100()
    # for p in t_model.parameters():
    #     print(p.shape)

    new_dic = deepcopy(t_model.state_dict())
    keys = list(new_dic.keys())
    j = 0
    for i in range(2, 6):
        new_dic[keys[2*j]] = torch.tensor(raw_weights[i])
        new_dic[keys[2 * j+1]] = torch.tensor(raw_bias[i])
        j += 1
    for i in range(0, 2):
        new_dic[keys[2*j]] = torch.tensor(raw_weights[i])
        new_dic[keys[2 * j+1]] = torch.tensor(raw_bias[i])
        j += 1

    t_model.load_state_dict(new_dic)

    # for ch in t_model.children():
    #     print(isinstance(ch, nn.Conv2d), isinstance(ch, nn.Linear))
    #
    # for e in t_model.named_children():
    #     print(e)
    # quit()

    true_count = 0
    # image = images[0:1]
    # image = np.transpose(image, (0, 3, 1, 2))
    # image = torch.tensor(image, dtype=torch.float)
    c0 = torch.tensor(c0)
    # print(c0.shape)
    # mean = torch.reshape(c0, [3, 1, 1])
    mean = c0[:, 0]
    # c0 = c0[:, 0]
    c2 = torch.tensor(c2)
    std = c2[:, 0]
    # std = torch.reshape(c2, [3, 1, 1])

    # coeff = torch.ones((2, 3, 2, 2))
    # const = torch.zeros(2)
    # sp = (torch.sum(coeff, dim=[2, 3])).shape
    # a = torch.tile(mean, (coeff.shape[0], 1)) * torch.sum(coeff, dim=[2, 3])
    # print(a / std)
    # print(torch.sum(coeff, dim=[2, 3]))
    # print((torch.sum(coeff, dim=[2, 3]) * mean).shape)
    # print(mean)
    # print(std)
    # print(mean / std * 4)
    # print(torch.sum(coeff, dim=[2, 3]) * mean / std)
    # quit()

    images = np.expand_dims(images, 1)
    # images = np.transpose(images, (0, 3, 1, 2))
    images = torch.tensor(images, dtype=torch.float)
    labels = torch.tensor(labels)

    for i in range(images.shape[0]):
        image = images[i:i+1]
        # image = torch.unsqueeze(image, dim=0)
        y = t_model((image - c0) / c2)
        pred = torch.argmax(y, dim=1).item()
        if pred == labels[i].item():
            true_count += 1

    print(true_count/images.shape[0])


    # torch.save(t_model.state_dict(), './mnist_6100.pth')
