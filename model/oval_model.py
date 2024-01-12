import pickle
import torch
import torch.nn as nn
from torch.nn import Flatten
import numpy as np
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from copy import deepcopy


class OVALBASE(nn.Module):
    def __init__(self):
        super(OVALBASE, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, (4, 4), stride=(2, 2), padding=(1, 1))
        self.linear1 = nn.Linear(1024, 100)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        y1 = nn.ReLU()(self.conv1(x))
        y2 = nn.ReLU()(self.conv2(y1))
        y3 = torch.flatten(y2, start_dim=1)
        y4 = nn.ReLU()(self.linear1(y3))
        y5 = self.linear2(y4)
        return y5


class OVALWIDE(nn.Module):
    def __init__(self):
        super(OVALWIDE, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, (4, 4), stride=(2, 2), padding=(1, 1))
        self.linear1 = nn.Linear(2048, 100)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        y1 = nn.ReLU()(self.conv1(x))
        y2 = nn.ReLU()(self.conv2(y1))
        y3 = torch.flatten(y2, start_dim=1)
        y4 = nn.ReLU()(self.linear1(y3))
        y5 = self.linear2(y4)
        return y5


class OVALDEEP(nn.Module):
    def __init__(self):
        super(OVALDEEP, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, (4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 8, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(8, 8, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(8, 8, (4, 4), stride=(2, 2), padding=(1, 1))
        self.linear1 = nn.Linear(512, 100)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        y1 = nn.ReLU()(self.conv1(x))
        y2 = nn.ReLU()(self.conv2(y1))
        y3 = nn.ReLU()(self.conv3(y2))
        y4 = nn.ReLU()(self.conv4(y3))
        y5 = torch.flatten(y4, start_dim=1)
        y6 = nn.ReLU()(self.linear1(y5))
        y7 = self.linear2(y6)
        return y7


def oval_base():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def test():
    std = [0.225, 0.225, 0.225]
    mean = [0.485, 0.456, 0.406]

    mean = torch.tensor(mean, dtype=torch.float)
    mean = torch.reshape(mean, [1, 3, 1, 1])
    std = torch.tensor(std, dtype=torch.float)
    std = torch.reshape(std, [1, 3, 1, 1])

    test_set = CIFAR10('../data/', train=False, download=False, transform=ToTensor())

    images_ori = np.transpose(test_set.data, (0, 3, 1, 2))
    # print((images_ori[0, 0] / 255 - 0.485) / 0.225)
    images_ori = torch.tensor(images_ori, dtype=torch.float) / 255
    images_ori = (images_ori - mean) / std
    t_ori = images_ori[0]
    # print(t_ori[0])
    # quit()
    a = torch.tensor(t_ori[1, :, :], dtype=torch.float)
    # a = (a - mean[0, 0, 0, 0]) / std[0, 0, 0, 0]
    print(a)

    normal = Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    test_set = CIFAR10('../data/', train=False, download=False, transform=Compose([ToTensor(), normal]))
    data, labels = zip(*test_set)
    images = torch.stack(data, dim=0)
    zd = images[0]
    # print(zd.shape)
    # print(zd[1, :, :])
    b = zd[1, :, :]
    print(b)

    # print((a == b).all())
    print(torch.max(torch.abs(a - b)))

    # print(images[0])
    # quit()

    # images = test_set.data / 255
    # labels = test_set.targets
    #
    # images = np.transpose(images, (0, 3, 1, 2))
    # images = torch.tensor(images, dtype=torch.float)
    quit()


if __name__ == '__main__':
    # test()
    path = './oval/base_100.pkl'

    with open(path, 'rb') as f_data:
        pd_data = pickle.load(f_data)

    np_data = pd_data.values
    idx = np_data[:, 0]
    eps_list = np_data[:, 1]
    specification = np_data[:, 2]

    test_set = CIFAR10('../data/', train=False, download=False, transform=ToTensor())
    images = test_set.data
    labels = test_set.targets

    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.tensor(images, dtype=torch.float) / 255

    # normal = Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    # test_set = CIFAR10('../data/', train=False, download=False, transform=Compose([ToTensor(), normal]))
    # data, labels = zip(*test_set)
    # images = torch.stack(data, dim=0)


    i = 0
    print(np_data.shape)
    print(eps_list[0])
    # print(idx[i])
    # print(labels[idx[i]])
    # print(specification[i])
    paras = torch.load('./oval/cifar_base1.pth')
    model = OVALBASE()

    t_state = deepcopy(model.state_dict())
    t_keys = list(t_state.keys())
    p_keys = list(paras.keys())
    for i in range(len(t_keys)):
        t_state[t_keys[i]] = paras[p_keys[i]]
    model.load_state_dict(t_state)

    # model = oval_base()
    # model.load_state_dict(paras)

    std = [0.225, 0.225, 0.225]
    mean = [0.485, 0.456, 0.406]

    mean = torch.tensor(mean, dtype=torch.float)
    mean = torch.reshape(mean, [1, 3, 1, 1])
    std = torch.tensor(std, dtype=torch.float)
    std = torch.reshape(std, [1, 3, 1, 1])
    images = (images - mean) / std

    # images = (torch.tensor(images, dtype=torch.float) - mean) / std
    # images = torch.tensor(images, dtype=torch.float)
    # images = torch.reshape(images, [-1, 3, 32, 32])
    # images = (images - mean) / std


    true_count = 0
    for i in range(100):
        # print(idx[i])
        image = images[idx[i]:idx[i]+1]
        # image = (image - mean) / std
        # print((image - mean) / std)
        # print(labels[idx[0]])
        # image = images[i:i + 1]
        # image = torch.unsqueeze(image, dim=0)
        y = model(image)
        # print(y)
        pred = torch.argmax(y, dim=1).item()
        # print(pred, labels[i])
        if pred == labels[idx[i]]:
            true_count += 1


    print(true_count)
    print(true_count / 100)

    # torch.save(model.state_dict(), './oval_para/ovalbase.pth')
    quit()


    info = torch.load('./oval/cifar_base.pth', map_location=torch.device('cpu'))
    print(info.keys())

    states = info['state_dict'][0]

    keys = list(states.keys())
    for k in keys:
        print(states[k].shape)
    print('-'*40)
    # model = OVALBASE()
    model = oval_base()
    for p in model.parameters():
        print(p.shape)

    img = images[0:1]
    img = torch.reshape(img, [1, 3, 32, 32])

    # print(model(img))
    # quit()
    print('*'*40)
    # new_dict = deepcopy(model.state_dict())
    # new_dict_keys = list(new_dict.keys())
    # for i in range(len(keys)):
    #     # print(new_dict_keys[i])
    #     new_dict[new_dict_keys[i]] = states[keys[i]]

    # model.load_state_dict(new_dict)
    model.load_state_dict(states)

    mean = [0.225, 0.225, 0.225]
    std = [0.485, 0.456, 0.406]

    mean = torch.tensor(mean, dtype=torch.float)
    mean = torch.reshape(mean, [3, 1, 1])
    std = torch.tensor(std, dtype=torch.float)
    std = torch.reshape(std, [3, 1, 1])

    true_count = 0
    for i in range(images.shape[0]):
        # print(idx[i])
        # image = images[idx[i]:idx[i]+1]
        image = images[i:i + 1]
        # image = torch.unsqueeze(image, dim=0)
        y = model(image)
        # print(y)
        pred = torch.argmax(y, dim=1).item()
        # print(pred, labels[i])
        if pred == labels[i]:
            true_count += 1

    print(true_count)
    print(true_count/images.shape[0])
