import torch


def relu_bound(lb, ub):
    lower_slops = torch.zeros_like(lb)
    lower_intercepts = torch.zeros_like(lb)
    upper_slops = torch.zeros_like(lb)
    upper_intercepts = torch.zeros_like(lb)

    l_pos = torch.where(ub + lb > 0)
    lower_slops[l_pos] = 1.0

    u_pos = torch.where(lb >= 0)
    upper_slops[u_pos] = 1.0
    und = torch.where(torch.logical_and(lb < 0, ub > 0))
    upper_slops[und] = ub[und] / (ub[und] - lb[und])
    upper_intercepts[und] = -1 * upper_slops[und] * lb[und]

    # pos_num = u_pos[0].numel()
    # neg_num = torch.where(ub <= 0)[0].numel()
    # und_num = und[0].numel()
    # print('pos: ', pos_num, ', neg: ', neg_num, ', und: ', und_num)
    # print(pos_num + neg_num + und_num)
    # print(und[0])
    # print(und[1])
    # print(und[2])


    return lower_slops, lower_intercepts, upper_slops, upper_intercepts


def relu_bound_wk(lb, ub):
    lower_slops = torch.zeros_like(lb)
    lower_intercepts = torch.zeros_like(lb)
    upper_slops = torch.zeros_like(lb)
    upper_intercepts = torch.zeros_like(lb)

    neg = torch.where(ub < 0)
    pos = torch.where(lb > 0)
    upper_slops[pos] = 1.0
    lower_slops[pos] = 1.0

    und = torch.where(torch.logical_and(lb < 0, ub > 0))
    k = ub[und] / (ub[und] - lb[und])
    lower_slops[und] = k
    upper_slops[und] = k
    upper_intercepts[und] = -1 * lb[und] * k

    return lower_slops, lower_intercepts, upper_slops, upper_intercepts


if __name__ == '__main__':
    ls = torch.tensor([-2, 0, -0.5, -1])
    us = torch.tensor([0, 2, 1, 0.5])
    a, b, c, d = relu_bound(ls, us)
    print(a, b, c, d, sep='\n')
    # print(ls)

