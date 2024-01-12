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


std = [0.225, 0.225, 0.225]
mean = [0.485, 0.456, 0.406]

# h_size = [[8, 16, 16], [16, 8, 8], [100]]
# h_size = [[16, 16, 16], [32, 8, 8], [100]]
h_size = [[8, 16, 16], [8, 16, 16], [8, 16, 16], [8, 8, 8], [100]]
in_size = [3, 32, 32]

# path = './model/oval/base_100.pkl'

# model = OVALBASE()
# model.load_state_dict(torch.load('./model/oval_para/ovalbase.pth'))
# model = OVALWIDE()
# model.load_state_dict(torch.load('./model/oval_para/ovalwide.pth'))
model = OVALDEEP()
model.load_state_dict(torch.load('./model/oval_para/ovaldeep.pth'))

# with open(path, 'rb') as f_data:
#     pd_data = pickle.load(f_data)
# np_data = pd_data.values

# np_data = np.load('./model/oval_para/base_100.npy', allow_pickle=True)
# np_data = np.load('./model/oval_para/wide_100.npy', allow_pickle=True)
np_data = np.load('./model/oval_para/deep_100.npy', allow_pickle=True)

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

device = 'cuda:0'
model.to(device)
images = images.to(device)

for i in tqdm(range(50, 100)):
    img = images[idx[i]:idx[i] + 1]
    l = int(labels[idx[i]])
    t = specification[i]
    eps = eps_list[i] * 0.225
    # epsilon need to be changed

    c = torch.zeros(10, dtype=torch.float, device=device)
    c[l] = 1.0
    c[t] = -1.0

    start_time = time()
    ans = bab_batch(vmodel, img, eps, c, tlimit=1800, batch_size=64, strategy='rand')
    end_time = time()
    print('##item start##')
    print(end_time - start_time)
    print(ans)
    vmodel.reset()
