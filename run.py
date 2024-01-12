import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from model.model import *
from convback import VMODEL
from bab import bab_batch
from tqdm import tqdm
from time import time
import numpy as np


# model = MNISTConv()
# model = MNISTConvMed()
model = M6100()
# model = M9200()

# model = C6100()
# model = C9200()
# model = CIFARConv()
# model = CIFARConvMed()

# weight = torch.load('./model/mnist_convsmall_weight.pth')
weight = torch.load('./model/mnist_6100.pth')
# weight = torch.load('./model/mnist_9200.pth')
# weight = torch.load('./model/mnist_convmed.pth')

# weight = torch.load('./model/cifar_6100.pth')
# weight = torch.load('./model/cifar_9200.pth')
# weight = torch.load('./model/cifar_convsmall.pth')
# weight = torch.load('./model/cifar_convmed.pth')
model.load_state_dict(weight)

# for mnist conv
# mean = [0.1307]
# std = [0.3081]

# for mnist and
mean = [0.0]
std = [1.0]

# for cifar fnn
# mean = [0.0, 0.0, 0.0]
# std = [1.0, 1.0, 1.0]

# for cifar and c9200
# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2023, 0.1994, 0.2010]

# h_size = [[16, 13, 13], [32, 5, 5], [100]]
# h_size = [[16, 14, 14], [32, 7, 7], [1000]]
h_size = [[100], [100], [100], [100], [100]]
# h_size = [[200], [200], [200], [200], [200], [200], [200], [200]]

# h_size = [[100], [100], [100], [100], [100], [100]]
# h_size = [[200], [200], [200], [200], [200], [200], [200], [200], [200]]
# h_size = [[16, 15, 15], [32, 6, 6], [100]]
# h_size = [[16, 16, 16], [32, 8, 8], [1000]]

in_size = [1, 28, 28]
# in_size = [3, 32, 32]

test_set = MNIST('./data/', train=False, download=False, transform=ToTensor())
# test_set = CIFAR10('./data/', train=False, download=False, transform=ToTensor())
images = test_set.data / 255
labels = test_set.targets

# images = np.transpose(images, (0, 3, 1, 2))
images = torch.tensor(images, dtype=torch.float)

device = 'cuda:0'
model.to(device)
images = images.to(device)

vmodel = VMODEL(model, mean, std, h_size, in_size)

rmean = torch.tensor(mean)
rmean = torch.reshape(rmean, [1, 1, 1]).to(device)
rstd = torch.tensor(std)
rstd = torch.reshape(rstd, [1, 1, 1]).to(device)

talk = 0
eps = 0.02
for i in tqdm(range(25)):
    img = images[i:i + 1]
    img = torch.reshape(img, (1, 1, 28, 28))
    # img = torch.reshape(img, (1, 3, 32, 32))
    l = int(labels[i])

    pred = vmodel.model((img - rmean) / rstd)
    if torch.argmax(pred).item() != l:
        continue
    # talk += 1
    # continue
    for j in range(10):
        if j == l:
            continue
        c = torch.zeros(10, dtype=torch.float, device=device)
        c[l] = 1.0
        c[j] = -1.0

        start_time = time()
        ans = bab_batch(vmodel, img, eps, c, tlimit=300, batch_size=64, strategy='lift')
        end_time = time()

        print('##item start##')
        print(end_time - start_time)
        print(ans)
        print(i, j)
        vmodel.reset()

# print(talk)
