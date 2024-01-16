import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from model.model import *
from model.oval_model import *
from convback import VMODEL
from bab import bab_batch
from tqdm import tqdm
from time import time
import pickle
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='choose a saved model')
parser.add_argument('--dataset', default='mnist', help='dataset')
parser.add_argument('--device', default='cuda:0', help='cpu or gpu')
parser.add_argument('--eps', default=0.015, help='radius')
parser.add_argument('--branch', default='optgbs', choices=['rand', 'babsr', 'fsb', 'optgbs'], help='branching strategy')
parser.add_argument('--batch_size', default=400, type=int, help='batch size')
parser.add_argument('--tlimit', default=1800, help='time limit for each property')
args = parser.parse_args()

net_name == args.model

if net_name == 'base':
    model = OVALBASE()
    model.load_state_dict(torch.load('./model/oval_para/ovalbase.pth'))
    std = [0.225, 0.225, 0.225]
    mean = [0.485, 0.456, 0.406]
    h_size = [[8, 16, 16], [16, 8, 8], [100]]
    np_data = np.load('./model/oval_para/base_100.npy', allow_pickle=True)
elif net_name == 'deep':
    model = OVALDEEP()
    model.load_state_dict(torch.load('./model/oval_para/ovaldeep.pth'))
    std = [0.225, 0.225, 0.225]
    mean = [0.485, 0.456, 0.406]
    h_size = [[8, 16, 16], [8, 16, 16], [8, 16, 16], [8, 8, 8], [100]]
    np_data = np.load('./model/oval_para/deep_100.npy', allow_pickle=True)
elif net_name == 'wide':
    model = OVALWIDE()
    model.load_state_dict(torch.load('./model/oval_para/ovalwide.pth'))
    std = [0.225, 0.225, 0.225]
    mean = [0.485, 0.456, 0.406]
    h_size = [[16, 16, 16], [32, 8, 8], [100]]
    np_data = np.load('./model/oval_para/wide_100.npy', allow_pickle=True)
else:
    print('Unimplemented model.')
    return 0

in_size = [3, 32, 32]

idx = np_data[:, 0]
idx = np.array(idx, dtype=np.int32)
eps_list = np_data[:, 1]
specification = np_data[:, 2]
specification = np.array(specification, dtype=np.int32)

test_set = CIFAR10('./data/', train=False, download=False, transform=ToTensor())
images = test_set.data
labels = test_set.targets
images = np.transpose(images, (0, 3, 1, 2))
images = torch.tensor(images, dtype=torch.float) / 255

vmodel = VMODEL(model, mean, std, h_size, in_size)

device = args.device
model.to(device)
images = images.to(device)

time_limit = args.tlimit
batch_size = args.batch_size
strategy = args.branch
if strategy == 'optgbs':
    strategy = 'lift'

for i in tqdm(range(100)):
    img = images[idx[i]:idx[i] + 1]
    l = int(labels[idx[i]])
    t = specification[i]
    eps = eps_list[i] * 0.225
    # epsilon need to be changed

    c = torch.zeros(10, dtype=torch.float, device=device)
    c[l] = 1.0
    c[t] = -1.0

    start_time = time()
    ans = bab_batch(vmodel, img, eps, c, tlimit=time_limit, batch_size=batch_size, strategy=strategy)
    end_time = time()
    print('##item start##')
    print(end_time - start_time)
    print(ans)
    vmodel.reset()
