import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from model.model import *
from convback import VMODEL
from bab import bab_batch
from tqdm import tqdm
from time import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='choose a saved model')
parser.add_argument('--dataset', default='mnist', help='dataset')
parser.add_argument('--device', default='cuda:0', help='cpu or gpu')
parser.add_argument('--eps', default=0.015, help='radius')
parser.add_argument('--branch', default='optgbs', choices=['rand', 'babsr', 'fsb', 'optgbs'], help='branching strategy')
parser.add_argument('--batch_size', default=400, type=int, help='batch size')
parser.add_argument('--tlimit', default=300, help='time limit for each property')
args = parser.parse_args()


net_name == args.model

support_model = ['m6100', 'm9200', 'mcnnsmall', 'mcnnbig', 'c6100', 'ccnnsmall', 'ccnnbig']

if net_name not in support_model:
    print('Unimplemented model.')
    return 0
else:
    model_idx = support_model.index(net_name)

if net_name == 'm6100':
    model = M6100()
    weight = torch.load('./model/mnist_6100.pth')
    mean = [0.0]
    std = [1.0]
    h_size = [[100], [100], [100], [100], [100]]
elif net_name == 'm9200':
    model = M9200()
    weight = torch.load('./model/mnist_9200.pth')
    mean = [0.0]
    std = [1.0]
    h_size = [[200], [200], [200], [200], [200], [200], [200], [200]]
elif net_name == 'mcnnsmall':
    model = MNISTConv()
    weight = torch.load('./model/mnist_convsmall_weight.pth')
    mean = [0.1307]
    std = [0.3081]
    h_size = [[16, 13, 13], [32, 5, 5], [100]]
elif net_name == 'mcnnbig':
    model = MNISTConvMed()
    weight = torch.load('./model/mnist_convmed.pth')
    mean = [0.1307]
    std = [0.3081]
    h_size = [[16, 14, 14], [32, 7, 7], [1000]]
elif net_name == 'c6100':
    model = C6100()
    weight = torch.load('./model/cifar_6100.pth')
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]
    h_size = [[100], [100], [100], [100], [100], [100]]
elif net_name == 'ccnnsmall':
    model = CIFARConv()
    weight = torch.load('./model/cifar_convsmall.pth')
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    h_size = [[16, 15, 15], [32, 6, 6], [100]]
elif net_name == 'ccnnbig':
    model = CIFARConvMed()
    weight = torch.load('./model/cifar_convmed.pth')
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    h_size = [[16, 16, 16], [32, 8, 8], [1000]]

model.load_state_dict(weight)

if model_idx < 4:
    in_size = [1, 28, 28]
    test_set = MNIST('./data/', train=False, download=False, transform=ToTensor())
    images = test_set.data / 255
    labels = test_set.targets
    images = torch.tensor(images, dtype=torch.float)
    rshape = [1, 1, 1]
else:
    in_size = [3, 32, 32]
    test_set = CIFAR10('./data/', train=False, download=False, transform=ToTensor())
    images = test_set.data / 255
    labels = test_set.targets
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.tensor(images, dtype=torch.float)
    rshape = [3, 1, 1]


device = args.device
model.to(device)
images = images.to(device)

vmodel = VMODEL(model, mean, std, h_size, in_size)

rmean = torch.tensor(mean)
rmean = torch.reshape(rmean, rshape).to(device)
rstd = torch.tensor(std)
rstd = torch.reshape(rstd, rshape).to(device)

eps = args.eps
time_limit = args.tlimit
batch_size = args.batch_size
strategy = args.branch
if strategy == 'optgbs':
    strategy = 'lift'

for i in tqdm(range(100)):
    img = images[i:i + 1]
    
    img = torch.reshape(img, [1] + in_size)
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
        ans = bab_batch(vmodel, img, eps, c, tlimit=time_limit, batch_size=batch_size, strategy=strategy)
        end_time = time()

        print('##item start##')
        print(end_time - start_time)
        print(ans)
        print(i, j)
        vmodel.reset()

# print(talk)
