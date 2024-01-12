import torch
from torch.utils.data import DataLoader

def merge(cover1, cover2):
    m_cover = cover1 + cover2
    m1 = torch.where(torch.logical_and(m_cover<2, m_cover>-2), torch.zeros_like(m_cover), m_cover)
    m1 = torch.where(m1 == 2, torch.ones_like(m1), m1)
    m1 = torch.where(m1 == -2, -1*torch.ones_like(m1), m1)

    return m1


def generate_sample(x, eps, num, norm='inf'):
    xmin = torch.clip(x - eps, min=0.0)
    xmax = torch.clip(x + eps, max=1.0)
    shape = list(x.shape)
    shape[0] = num
    pert = torch.rand(shape)
    samples = xmin + pert * (xmax - xmin)

    return samples


def find_bi_coverage(data, eps, sample_num, model, mean, std, batch_size=128, norm='inf'):
    # only consider activate or deactivate for relu
    samples = generate_sample(data, eps, sample_num)

    sample_loader = DataLoader(samples, batch_size=batch_size)
    coverage = []
    flag = False
    for samps in sample_loader:
        trace = model.forward_detail((samps - mean) / std)
        for i in range(len(trace)):
            t_num = trace[i].shape[0]
            t = torch.where(trace[i] > 0, 1, -1)
            t1 = torch.sum(t, dim=[0])
            # print(t1[0])
            t2 = torch.where(torch.logical_and(t1 < t_num, t1 > -1 * t_num), torch.zeros_like(t1), t1)
            t2 = torch.where(t2 == t_num, torch.ones_like(t2), t2)
            t2 = torch.where(t2 == -1*t_num, -1*torch.ones_like(t2), t2)

            if flag:
                coverage[i] = merge(coverage[i], t2)
            else:
                coverage.append(t2)
        flag = True
    for i in range(len(coverage)):
        und = torch.where(coverage[i] == 0)[0].numel()
        pos = torch.where(coverage[i] == 1)[0].numel()
        neg = torch.where(coverage[i] == -1)[0].numel()
        print('pos: ', pos, 'neg: ', neg, 'und: ', und)
        if i == 2:
            idx = torch.where(coverage[i] == 0)
            print(idx)


def find_bi_coverage_sample(samples, model, mean, std, batch_size=128, norm='inf'):
    sample_loader = DataLoader(samples, batch_size=batch_size)
    coverage = []
    flag = False
    for samps in sample_loader:
        trace = model.forward_detail((samps - mean) / std)
        for i in range(len(trace)):
            t_num = trace[i].shape[0]
            t = torch.where(trace[i] > 0, 1, -1)
            t1 = torch.sum(t, dim=[0])
            # print(t1[0])
            t2 = torch.where(torch.logical_and(t1 < t_num, t1 > -1 * t_num), torch.zeros_like(t1), t1)
            t2 = torch.where(t2 == t_num, torch.ones_like(t2), t2)
            t2 = torch.where(t2 == -1 * t_num, -1 * torch.ones_like(t2), t2)

            if flag:
                coverage[i] = merge(coverage[i], t2)
            else:
                coverage.append(t2)
        flag = True
    for i in range(len(coverage)):
        und = torch.where(coverage[i] == 0)[0].numel()
        pos = torch.where(coverage[i] == 1)[0].numel()
        neg = torch.where(coverage[i] == -1)[0].numel()
        print('pos: ', pos, 'neg: ', neg, 'und: ', und)
        # und = torch.where(coverage[i] == 0)
        # print(und[0])
        # print(und[1])
        # print(und[2])


if __name__ == '__main__':
    # torch.random.manual_seed(1)
    torch.set_printoptions(precision=8, linewidth=400)
    from model.model import *
    model = MNISTConv()
    weight = torch.load('./model/mnist_convsmall_weight.pth')
    model.load_state_dict(weight)
    mean = 0.1207
    std = 0.3081

    from torchvision.datasets import MNIST, CIFAR10
    from torchvision.transforms import ToTensor

    test_set = MNIST('./data/', train=False, download=False, transform=ToTensor())
    # test_set = CIFAR10('./data/', train=False, download=False, transform=ToTensor())
    images = test_set.data / 255
    labels = test_set.targets

    x = images[0]
    x = torch.reshape(x, [1, 1, 28, 28])

    find_bi_coverage(x, 0.1, 10000, model, mean, std)
