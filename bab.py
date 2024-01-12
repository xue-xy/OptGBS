from convback import VMODEL
from domain import *
from bouding import verify_domain_beta, verify_domain_beta_choice
from branching import babsr_single, fsb_single, optimal_lift, babsr_batch, fsb_batch, rc_single, rc_batch
import torch
from time import time
from attack import pgd_attack


def bab(vmodel: VMODEL, x, eps, c, batch_size=4, tlimit=10, norm='inf'):
    original_val, o_pce = vmodel.backward_propagation(x, eps, c, norm=norm)
    # print(original_val)
    # print(vmodel.check(o_pce, c))
    if original_val >= 0:
        return 'yes', 0
    elif vmodel.check(o_pce, c) < 0:
        return 'no', 0

    ori_dom = DOMAIN(vmodel.layer_num - 1)
    unsat_doms = LIST([(original_val, ori_dom)])
    checked_num = 0

    bab_start_time = time()
    dom_time = 0
    bound_time = 0
    post_time = 0

    while len(unsat_doms) > 0:

        d_start = time()

        check_dom = []
        for i in range(min(batch_size // 2, len(unsat_doms))):
            dom = unsat_doms.get()[1]
            # print(dom)

            # choice, _ = babsr_single(vmodel, dom, c)
            choice, _ = fsb_single(vmodel, dom, x, eps, c)
            # choice, _ = optimal_lift(vmodel, dom, x, eps, c)

            # idx = torch.where(torch.logical_and(vmodel.hidden_bounds[2][0] < 0, vmodel.hidden_bounds[2][1] > 0))
            # forced_idx = [idx[i][:1] for i in range(len(idx))]
            # print(forced_idx)
            # choice = (2, forced_idx)
            check_dom.extend(split_single(dom, choice))

        dom_time += time() - d_start

        mb_start = time()
        dom_batch = make_domain_batch(check_dom)
        # print(dom_batch)
        makebatch_time = time() - mb_start

        b_start = time()

        dval, dpce = verify_domain_beta(vmodel, dom_batch, x, eps, c)
        # print(dval)
        # print('*'*70)

        bound_time += time() - b_start

        p_start = time()

        checked_num += len(check_dom)

        if (vmodel.check(dpce, c) < 0).any():
            return 'no', time() - bab_start_time, checked_num

        unsat_idx = torch.where(dval < 0)[0].tolist()
        for eid in unsat_idx:
            unsat_doms.push((dval[eid], check_dom[eid]))

        if (time() - bab_start_time) > tlimit:
            break

        post_time += time() - p_start
    time_cost = time() - bab_start_time

    print(dom_time, makebatch_time, bound_time, post_time)
    if len(unsat_doms) == 0:
        return 'yes', time_cost, checked_num
    else:
        # print(len(unsat_doms))
        return 'und', time_cost, checked_num


def bab_batch(vmodel: VMODEL, x, eps, c, b=torch.tensor(0.0), batch_size=2, tlimit=10, strategy='lift', norm='inf'):
    found = pgd_attack(vmodel, x, c, eps)
    if found:
        return 'no', 0

    original_val, o_pce = vmodel.backward_propagation(x, eps, c, b=b, norm=norm)
    # print(original_val)
    # print(vmodel.check(o_pce, c))
    if original_val >= 0:
        return 'yes', 0
    elif vmodel.check(o_pce, c) < 0:
        return 'no', 0

    ori_dom = DOMAIN(vmodel.layer_num - 1)

    if strategy == 'lift':
        choice, _ = optimal_lift(vmodel, ori_dom, x, eps, c)
    elif strategy == 'babsr':
        choice, _ = babsr_single(vmodel, ori_dom, c)
    elif strategy == 'fsb':
        choice, _ = fsb_single(vmodel, ori_dom, x, eps, c)
    elif strategy == 'rand':
        choice, _ = rc_single(vmodel, ori_dom, c)
    else:
        return 'strategy not contained'

    unsat_doms = LIST([(original_val, ori_dom, choice)])
    searched_num = 0

    bab_start_time = time()
    # dom_time = 0
    # bound_time = 0
    # post_time = 0
    # batch_make_time = 0

    while len(unsat_doms) > 0:

        # d_start = time()
        check_dom = []
        for i in range(min(batch_size // 2, len(unsat_doms))):
            toval, dom, choice = unsat_doms.get()

            check_dom.extend(split_single(dom, choice))
        # dom_time += time() - d_start

        # bm_start = time()
        dom_batch = make_domain_batch(check_dom)
        # batch_make_time += time() - bm_start

        # print(dom_batch)

        # b_start = time()
        if strategy == 'lift':
            dval, dpce, dchoices = verify_domain_beta_choice(vmodel, dom_batch, x, eps, c)
        elif strategy == 'babsr':
            dval, dpce = verify_domain_beta(vmodel, dom_batch, x, eps, c)
            dchoices, _ = babsr_batch(vmodel, dom_batch, c)
        elif strategy == 'fsb':
            dval, dpce = verify_domain_beta(vmodel, dom_batch, x, eps, c)
            dchoices, _ = fsb_batch(vmodel, dom_batch, check_dom, x, eps, c)
        elif strategy == 'rand':
            dval, dpce = verify_domain_beta(vmodel, dom_batch, x, eps, c)
            dchoices, _ = rc_batch(vmodel, dom_batch, x, eps, c)
        # bound_time += time() - b_start

        # p_start = time()

        searched_num += len(check_dom)

        if (vmodel.check(dpce, c) < 0).any():
            return 'no', time() - bab_start_time, searched_num

        unsat_idx = torch.where(dval < 0)[0].tolist()
        for eid in unsat_idx:
            unsat_doms.push((dval[eid], check_dom[eid], dchoices[eid]))

        # post_time += time() - p_start

        if (time() - bab_start_time) > tlimit:
            break
    time_cost = time() - bab_start_time
    # print(dom_time, batch_make_time, bound_time, post_time)
    if len(unsat_doms) == 0:
        return 'yes', time_cost, searched_num
    else:
        return 'und', time_cost, searched_num


if __name__ == '__main__':

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

    from torchvision.datasets import MNIST, CIFAR10
    from torchvision.transforms import ToTensor

    test_set = MNIST('./data/', train=False, download=False, transform=ToTensor())
    # test_set = CIFAR10('./data/', train=False, download=False, transform=ToTensor())
    images = test_set.data / 255
    labels = test_set.targets

    i = 2

    img = torch.reshape(images[i], (1, 1, 28, 28))

    # for ch in model.children():
    #     if isinstance(ch, nn.Conv2d):
    #         print(ch.weight.shape)
    #         print(ch.bias.shape)
    #         print(ch.dilation)
    #         print(ch.stride)
    # quit()

    vmodel = VMODEL(model, mean, std, h_size, in_size)
    c = torch.zeros(10, dtype=torch.float)
    l = labels[i].item()
    c[l] = 1.0
    if l == 0:
        c[1] = -1.0
    else:
        c[0] = -1.0

    # dom = DOMAIN(vmodel.layer_num - 1)
    #
    # idx = torch.where(torch.logical_and(vmodel.hidden_bounds[-1][0] < 0, vmodel.hidden_bounds[-1][1] > 0))
    # force_id = [idx[i][:1] for i in range(len(idx))]
    # n_id_pos = [torch.tensor([0])]
    # n_id_pos.extend(force_id)
    # n_id_neg = [torch.tensor([1])]
    # n_id_neg.extend([idx[i][1:2] for i in range(len(idx))])
    # dom.pos_id[-1] = n_id_pos
    # dom.neg_id[-1] = n_id_neg
    # dom.batch_size = 2
    # vmodel.backward_propagation(img, 0.15, c)
    # a = torch.argmin(vmodel.hidden_bounds[1][0] - vmodel.hidden_bounds[1][1])
    # b = torch.argmax(vmodel.hidden_bounds[1][0] - vmodel.hidden_bounds[1][1])
    # c = vmodel.hidden_bounds[1][0] - vmodel.hidden_bounds[1][1]
    # a = a.unsqueeze(0)
    # b = b.unsqueeze(0)
    # print(a, b)
    # print(vmodel.hidden_bounds[-1][0][a], vmodel.hidden_bounds[-1][1][a])
    # torch.cat([a, b])
    # print(c)
    ans = bab(vmodel, img, 0.13, c)
    print(ans)
