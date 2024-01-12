import torch
import torch.nn as nn
from torch.nn.functional import conv_transpose2d
from util import *
from activation import relu_bound
from copy import deepcopy
from coverage import find_bi_coverage_sample


class VMODEL:
    def __init__(self, model, mean, std, hidden_size, input_size, device='cuda:0'):
        self.model = model
        for _, p in model.named_parameters():
            p.requires_grad = False

        paras = model.state_dict()
        p_keys = list(paras.keys())
        self.layer_num = len(p_keys) // 2
        self.weights = [paras[p_keys[2 * i]].to(device) for i in range(self.layer_num)]
        self.biases = [paras[p_keys[2 * i + 1]].to(device) for i in range(self.layer_num)]
        self.children = list(model.children())

        self.mean = torch.tensor(mean).to(device)
        self.std = torch.tensor(std).to(device)
        self.mean_l = torch.reshape(self.mean, [len(mean), 1, 1])
        self.std_l = torch.reshape(self.std, [len(std), 1, 1])
        self.hidden_bounds = [None] * (self.layer_num - 1)              # each element is (lb, ub)
        self.saved_slops = [None] * (self.layer_num - 1)
        self.saved_intercepts = [None] * (self.layer_num - 1)
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hpce_set = torch.zeros([0]+input_size)

    def reset(self):
        self.hidden_bounds = [None] * (self.layer_num - 1)
        self.saved_slops = [None] * (self.layer_num - 1)
        self.saved_intercepts = [None] * (self.layer_num - 1)
        self.hpce_set = torch.zeros([0] + self.input_size)

    def check(self, pce, c):
        if pce.dim() == 3:
            pce = pce.unsqueeze(dim=0)
        pred = self.model((pce - self.mean_l) / self.std_l)

        if pred.dim() == 2:
            return torch.sum(c * pred.detach(), dim=1)
        else:
            return torch.sum(c * pred.detach())

    def weight_back(self, coeff, const, layer_id):
        if coeff.dim() == 2:
            const = torch.matmul(coeff, self.children[layer_id].bias) + const
            coeff = torch.matmul(coeff, self.children[layer_id].weight)
        elif coeff.dim() == 4:
            module = self.children[layer_id]

            coeff_sum = torch.sum(coeff, dim=[2, 3])
            const = torch.matmul(coeff_sum, module.bias) + const
            coeff = conv_transpose2d(coeff, module.weight, stride=module.stride, padding=module.padding,
                                     dilation=module.dilation)
            _, _, height, width = coeff.shape
            if layer_id == 0:
                coeff = torch.nn.functional.pad(coeff, (0, self.input_size[-1]-width, 0, self.input_size[-2]-height))
            else:
                coeff = torch.nn.functional.pad(coeff, (0, self.hidden_size[layer_id-1][-1] - width,
                                                        0, self.hidden_size[layer_id-1][-2] - height))
        return coeff, const

    def hidden_layers_bounds(self, x, eps):
        samples = torch.zeros([0]+self.input_size, device=self.device)
        for i in range(self.layer_num - 1):
            nele = torch.prod(torch.tensor(self.hidden_size[i])).item()
            coeff = torch.ones(nele, dtype=torch.float)
            shape = deepcopy(self.hidden_size[i])
            shape.insert(0, nele)

            lower_coeff = torch.reshape(torch.diag(coeff).to(self.device), shape)
            lower_const = torch.zeros(nele, dtype=torch.float, device=self.device)
            upper_coeff = torch.reshape(torch.diag(coeff).to(self.device), shape)
            upper_const = torch.zeros(nele, dtype=torch.float, device=self.device)

            lower_coeff, lower_const = self.weight_back(lower_coeff, lower_const, layer_id=i)
            upper_coeff, upper_const = self.weight_back(upper_coeff, upper_const, layer_id=i)

            # print(i)

            # if i == 2:
            #     quit()

            for j in range(i - 1, -1, -1):
                if lower_coeff.dim()-1 < len(self.hidden_size[j]):
                    shape_change = [lower_coeff.shape[0]]
                    shape_change.extend(self.hidden_size[j])
                    lower_coeff = torch.reshape(lower_coeff, shape_change)
                    upper_coeff = torch.reshape(upper_coeff, shape_change)

                inc_lower_const = torch.clip(lower_coeff, min=0) * self.saved_intercepts[j][0] + torch.clip(lower_coeff, max=0) * self.saved_intercepts[j][1]
                lower_const += torch.sum(inc_lower_const, dim=[d for d in range(1, inc_lower_const.dim())])
                lower_coeff = torch.clip(lower_coeff, min=0) * self.saved_slops[j][0] + torch.clip(lower_coeff, max=0) * self.saved_slops[j][1]
                inc_upper_const = torch.clip(upper_coeff, min=0) * self.saved_intercepts[j][1] + torch.clip(upper_coeff, max=0) * self.saved_intercepts[j][0]
                upper_const += torch.sum(inc_upper_const, dim=[d for d in range(1, inc_lower_const.dim())])
                upper_coeff = torch.clip(upper_coeff, min=0) * self.saved_slops[j][1] + torch.clip(upper_coeff, max=0) * self.saved_slops[j][0]

                lower_coeff, lower_const = self.weight_back(lower_coeff, lower_const, layer_id=j)
                upper_coeff, upper_const = self.weight_back(upper_coeff, upper_const, layer_id=j)

                # print(upper_coeff.shape, lower_coeff.shape)
                # print(upper_const.shape, upper_const.shape)

            # pay attention if mean and std are multi-dimensional

            # xmin = (torch.clip(x - eps, min=0) - self.mean) / self.std
            # xmax = (torch.clip(x + eps, max=1) - self.mean) / self.std
            # print(xmin[0, 0, :4, :4])
            # print((torch.clip(x + eps, max=1))[0, 0, :4, :4])
            # t = torch.reshape(lower_coeff[0, 0], (1, 1, 28, 28))
            # d = lower_const[0]
            # lb_coeff, _ = evaluate_inf_min_arg(xmin, xmax, t)
            # print(lb_coeff + d)
            # print('-'*50)
            # inc = t[0, 0, :4, :4] * self.mean
            # incp = torch.sum(inc) / self.std
            # print(incp)
            # tp = t / self.std
            # print(t[0, 0, :4, :4])
            # # print(tp[0, 0, :4, :4])
            #
            # xmin = torch.clip(x - eps, min=0)
            # xmax = torch.clip(x + eps, max=1)
            # val, _ = evaluate_inf_min_arg(xmin, xmax, tp)
            # quit()

            if lower_coeff.dim() - 1 < len(self.input_size):
                shape_change = [lower_coeff.shape[0]]
                shape_change.extend(self.input_size)
                lower_coeff = torch.reshape(lower_coeff, shape_change)
                upper_coeff = torch.reshape(upper_coeff, shape_change)

            lower_const = lower_const - torch.sum(torch.sum(lower_coeff, dim=[2, 3]) * self.mean / self.std, dim=1)  # reduced dim: 2 ... last
            lower_coeff = lower_coeff / self.std_l
            upper_const = upper_const - torch.sum(torch.sum(upper_coeff, dim=[2, 3]) * self.mean / self.std, dim=1)
            upper_coeff = upper_coeff / self.std_l
            xmin = torch.clip(x - eps, min=0)
            xmax = torch.clip(x + eps, max=1)

            # xmin = (torch.clip(x - eps, min=0) - self.mean) / self.std
            # xmax = (torch.clip(x + eps, max=1) - self.mean) / self.std

            lb_coeff, l_sample = evaluate_inf_min_arg(xmin, xmax, lower_coeff)
            lb_row = lb_coeff + lower_const
            lb = torch.reshape(lb_row, self.hidden_size[i])
            ub_coeff, u_sample = evaluate_inf_max_arg(xmin, xmax, upper_coeff)
            ub_row = ub_coeff + upper_const
            ub = torch.reshape(ub_row, self.hidden_size[i])

            # if i == 2:
            #     # lb_coeff, l_pce = evaluate_inf_min_arg(xmin, xmax, lower_coeff)
            #     # lb = lb_coeff + lower_const
            #     # ub_coeff, u_pce = evaluate_inf_max_arg(xmin, xmax, upper_coeff)
            #     # ub = ub_coeff + upper_const
            #
            #     # st = self.model.forward_detail((u_pce[idx] - self.mean) / self.std)
            #
                # und = torch.where(torch.logical_and(lb_row < 0, ub_row > 0))[0]
                # pce = torch.cat([u_sample[58:59], l_sample[58:59]])
                # trace = self.model.forward_detail((pce - self.mean) / self.std)
                # print(trace[2][:, 58])

            #     print(idx.numel())
            #
            #     pce = torch.cat([l_sample[idx], u_sample[idx]], dim=0)
            #     print(pce.shape)
            #     from coverage import find_bi_coverage_sample
            #     # trace = self.model.forward_detail((pce - self.mean) / self.std)
            #     # tt = trace[0]
            #     # print(torch.where(tt[:, 0, 2, 4] < 0))
            #     cover = find_bi_coverage_sample(pce, self.model, self.mean, self.std)
            #
            #     quit()
            #
            #     trace = self.model.forward_detail(u_pce[idx])
            #     layer1 = torch.flatten(trace[0], start_dim=1)
            #     print(layer1.shape)
            #     flag = True
            #     for i in range(682):
            #         if layer1[i, idx[i]] != ub[idx[i]]:
            #             flag = False
            #             print(layer1[i, idx[i]] - ub[idx[i]])
            #             # print(lb[idx[i]])
            #     print(flag)
            #     print(layer1[0, idx[0]])
            #     print(ub[idx[0]])
            #
            #     quit()

            self.hidden_bounds[i] = (lb, ub)

            lower_slop, lower_intercept, upper_slop, upper_intercept = relu_bound(lb, ub)

            self.saved_slops[i] = (lower_slop, upper_slop)
            self.saved_intercepts[i] = (lower_intercept, upper_intercept)

            # und = torch.where(torch.logical_and(lb_row < 0, ub_row > 0))[0]
            # samples = torch.cat([samples, l_sample[und], u_sample[und]], dim=0)
            # self.hpce_set = samples

            # module = self.children[0]
            # print(module.weight[0])
            # h1 = torch.nn.functional.conv2d(x, module.weight, module.bias, module.stride)[0].detach()
            # idx = torch.where(h1 < lb)
            # print(idx)

            # a, b = self.hidden_bounds[0]
            # print(a.shape, b.shape)

            # if i == 3:
            #     print(1)
            #     quit()
        return samples

    def backward_propagation(self, x, eps, c, b=torch.tensor(0.0), norm='inf'):
        # find the lower bound of the minimum value

        if self.hidden_bounds[0] is None:
            samples = self.hidden_layers_bounds(x, eps)
        # cover = find_bi_coverage_sample(samples, self.model, self.mean, self.std)
        # quit()

        if c.dim() == 1:
            c = torch.unsqueeze(c, 0)
        b.to(self.device)

        coeff, const = self.weight_back(c, b, layer_id=len(self.children)-1)

        for i in range(self.layer_num - 2, -1, -1):
            if coeff.dim()-1 < len(self.hidden_size[i]):
                shape_change = [coeff.shape[0]]
                shape_change.extend(self.hidden_size[i])
                coeff = torch.reshape(coeff, shape_change)
            inc_const = torch.clip(coeff, min=0) * self.saved_intercepts[i][0] + torch.clip(coeff, max=0) * self.saved_intercepts[i][1]
            const += torch.sum(inc_const, dim=[d for d in range(1, inc_const.dim())])
            coeff = torch.clip(coeff, min=0) * self.saved_slops[i][0] + torch.clip(coeff, max=0) * self.saved_slops[i][1]

            coeff, const = self.weight_back(coeff, const, layer_id=i)
            # print(const)

        if coeff.dim() - 1 < len(self.input_size):
            shape_change = [coeff.shape[0]]
            shape_change.extend(self.input_size)
            coeff = torch.reshape(coeff, shape_change)

        const = const - torch.sum(torch.sum(coeff, dim=[2, 3]) * self.mean / self.std, dim=1)  # reduced dim: 2 ... last
        coeff = coeff / self.std_l
        # pce for pesudo-counterexample
        xmin = torch.clip(x - eps, min=0)
        xmax = torch.clip(x + eps, max=1)
        val, pce = evaluate_inf_min_arg(xmin, xmax, coeff)
        # print(val, const)
        val = val + const
        # print(val)
        # print(self.check(pce, c))

        return val, pce


if __name__ == '__main__':

    # coeff = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
    # coeff = torch.reshape(coeff, [1, 1, 2, 2])
    # const = torch.tensor(0.0)
    # mean = 1.0
    # std = 0.5
    #
    # # const = const - torch.sum(torch.sum(coeff, dim=[2, 3]) * mean / std, dim=1)  # reduced dim: 2 ... last
    # # coeff = coeff / std
    # #
    # # print(const)
    # # print(coeff)
    #
    # xmin = torch.zeros((1, 1, 2, 2))
    # xmax = torch.ones((1, 1, 2, 2))
    #
    # xmin = (xmin - mean) / std
    # xmax = (xmax - mean) / std
    # # print(xmin)
    #
    # val, pce = evaluate_inf_min_arg(xmin, xmax, coeff)
    # print(val+const)
    #
    # quit()

    torch.set_printoptions(precision=8, linewidth=400)
    from model.model import MNISTConv
    model = MNISTConv()
    weight = torch.load('./model/mnist_convsmall_weight.pth')
    model.load_state_dict(weight)
    mean = 0.1207
    std = 0.3081
    # mean = 0.0
    # std = 1.0
    h_size = [[16, 13, 13], [32, 5, 5], [100]]
    in_size = [1, 28, 28]

    hs = [2, 3, 3]
    # nele = torch.prod(torch.tensor(hs)).item()
    # diag = torch.ones(nele)
    # mask = torch.diag(diag)
    # shape = deepcopy(hs)
    # shape.insert(0, nele)
    # print(shape)
    # mask = torch.reshape(mask, shape)
    # print(mask)
    # print(hs)
    #
    # quit()

    # a = torch.randn((2, 2, 3, 3))
    # b = torch.tensor([2, 3])
    # c = torch.sum(a, dim=[i for i in range(2, len(hs)+1)])
    # print(a.dim())
    # print(b * c)
    # quit()

    from torchvision.datasets import MNIST, CIFAR10
    from torchvision.transforms import ToTensor

    test_set = MNIST('./data/', train=False, download=False, transform=ToTensor())
    # test_set = CIFAR10('./data/', train=False, download=False, transform=ToTensor())
    images = test_set.data / 255
    labels = test_set.targets

    i = 0

    img = torch.reshape(images[i], (1, 1, 28, 28))
    # print(labels[1])

    # for ch in model.children():
    #     if isinstance(ch, nn.Conv2d):
    #         print(ch.weight.shape)
    #         print(ch.bias.shape)
    #         print(ch.dilation)
    #         print(ch.stride)
    # quit()


    vmodel = VMODEL(model, mean, std, h_size, in_size)
    c = torch.tensor([-1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
    vmodel.backward_propagation(img, 0.1, c)

    quit()

    true_count = 0

    for i in range(images.shape[0]):
        image = images[i:i + 1]
        image = torch.unsqueeze(image, dim=0)
        y = model(image)
        pred = torch.argmax(y, dim=1).item()
        if pred == labels[i].item():
            true_count += 1

    print(true_count / images.shape[0])
