from convback import VMODEL
from domain import DOMAIN
from util import evaluate_inf_min_arg
from activation import relu_bound_wk
import torch
import torch.optim as opt
import numpy as np
from copy import deepcopy


def recover_id(idxes, shape):
    lth = len(shape)
    s_tensor = torch.tensor(shape, dtype=torch.int)
    res = idxes
    rec_id = []
    for i in range(lth-1):
        modu = torch.prod(s_tensor[i+1:])
        n = res // modu
        rec_id.append(n)
        res = res % modu
        # print(res)
    rec_id.append(res)
    # print(rec_id)
    return rec_id


def verify_domain_simple(vmodel: VMODEL, domains: DOMAIN, x, eps, c, b=torch.tensor([0.0])):
    this_slops = deepcopy(vmodel.saved_slops)
    this_intercepts = deepcopy(vmodel.saved_intercepts)
    batch_size = domains.batch_size

    for i in range(vmodel.layer_num - 1):
        r_shape = [batch_size]
        r_shape.extend([1 for i in range(this_slops[i][0].dim())])
        this_slops[i] = list(this_slops[i])
        this_intercepts[i] = list(this_intercepts[i])
        this_slops[i][0] = this_slops[i][0].repeat(r_shape)
        this_slops[i][1] = this_slops[i][1].repeat(r_shape)
        this_intercepts[i][0] = this_intercepts[i][0].repeat(r_shape)
        this_intercepts[i][1] = this_intercepts[i][1].repeat(r_shape)
        # print(this_slops[i][0].shape)
        # print(this_intercepts[i][0].shape)

        if not (domains.pos_id[i] is None):
            this_slops[i][0][domains.pos_id[i]] = 1.0
            this_slops[i][1][domains.pos_id[i]] = 1.0
            this_intercepts[i][0][domains.pos_id[i]] = 0.0
            this_intercepts[i][1][domains.pos_id[i]] = 0.0
        if not (domains.neg_id[i] is None):
            # print(this_slops[i][0][domains.neg_id[i]])
            # print(this_slops[i][1][domains.neg_id[i]])
            # print(this_intercepts[i][0][domains.neg_id[i]])
            # print(this_intercepts[i][1][domains.neg_id[i]])
            # print(vmodel.hidden_bounds[i][0][domains.neg_id[i][1:]])
            # print(vmodel.hidden_bounds[i][1][domains.neg_id[i][1:]])
            this_slops[i][0][domains.neg_id[i]] = 0.0
            this_slops[i][1][domains.neg_id[i]] = 0.0
            this_intercepts[i][0][domains.neg_id[i]] = 0.0
            this_intercepts[i][1][domains.neg_id[i]] = 0.0

    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)
    c = c.repeat(batch_size, 1)
    b = b.to(vmodel.device)

    coeff, const = vmodel.weight_back(c, b, layer_id=len(vmodel.children) - 1)

    for i in range(vmodel.layer_num - 2, -1, -1):
        if coeff.dim() - 1 < len(vmodel.hidden_size[i]):
            shape_change = [coeff.shape[0]]
            shape_change.extend(vmodel.hidden_size[i])
            coeff = torch.reshape(coeff, shape_change)
        inc_const = torch.clip(coeff, min=0) * this_intercepts[i][0] + torch.clip(coeff, max=0) * \
                    this_intercepts[i][1]
        const += torch.sum(inc_const, dim=[d for d in range(1, inc_const.dim())])
        coeff = torch.clip(coeff, min=0) * this_slops[i][0] + torch.clip(coeff, max=0) * this_slops[i][1]

        coeff, const = vmodel.weight_back(coeff, const, layer_id=i)
        # print(const)

    if coeff.dim() - 1 < len(vmodel.input_size):
        shape_change = [coeff.shape[0]]
        shape_change.extend(vmodel.input_size)
        coeff = torch.reshape(coeff, shape_change)

    const = const - torch.sum(torch.sum(coeff, dim=[2, 3]) * vmodel.mean / vmodel.std, dim=1)  # reduced dim: 2 ... last
    coeff = coeff / vmodel.std_l
    # pce for pesudo-counterexample
    xmin = torch.clip(x - eps, min=0)
    xmax = torch.clip(x + eps, max=1)
    val, pce = evaluate_inf_min_arg(xmin, xmax, coeff)
    # print(val, const)
    val = val + const
    # print(val)
    # print(pce.shape)
    # print(vmodel.check(pce, c))
    return val, pce


def verify_domain_simple_wk(vmodel: VMODEL, domains: DOMAIN, x, eps, c, b=torch.tensor([0.0])):
    this_slops = []
    this_intercepts = []
    batch_size = domains.batch_size

    for i in range(vmodel.layer_num - 1):
        lower_slop, lower_intercept, upper_slop, upper_intercept = relu_bound_wk(vmodel.hidden_bounds[i][0], vmodel.hidden_bounds[i][1])

        r_shape = [batch_size]
        r_shape.extend([1 for i in range(lower_slop.dim())])
        this_slops.append((lower_slop.repeat(r_shape), upper_slop.repeat(r_shape)))
        this_intercepts.append((lower_intercept.repeat(r_shape), upper_intercept.repeat(r_shape)))
        # print(this_slops[i][0].shape)
        # print(this_intercepts[i][0].shape)

        if not (domains.pos_id[i] is None):
            this_slops[i][0][domains.pos_id[i]] = 1.0
            this_slops[i][1][domains.pos_id[i]] = 1.0
            this_intercepts[i][0][domains.pos_id[i]] = 0.0
            this_intercepts[i][1][domains.pos_id[i]] = 0.0
        if not (domains.neg_id[i] is None):
            this_slops[i][0][domains.neg_id[i]] = 0.0
            this_slops[i][1][domains.neg_id[i]] = 0.0
            this_intercepts[i][0][domains.neg_id[i]] = 0.0
            this_intercepts[i][1][domains.neg_id[i]] = 0.0

    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)
    c = c.repeat(batch_size, 1)
    b = b.to(vmodel.device)

    coeff, const = vmodel.weight_back(c, b, layer_id=len(vmodel.children) - 1)

    for i in range(vmodel.layer_num - 2, -1, -1):
        if coeff.dim() - 1 < len(vmodel.hidden_size[i]):
            shape_change = [coeff.shape[0]]
            shape_change.extend(vmodel.hidden_size[i])
            coeff = torch.reshape(coeff, shape_change)
        inc_const = torch.clip(coeff, min=0) * this_intercepts[i][0] + torch.clip(coeff, max=0) * \
                    this_intercepts[i][1]
        const += torch.sum(inc_const, dim=[d for d in range(1, inc_const.dim())])
        coeff = torch.clip(coeff, min=0) * this_slops[i][0] + torch.clip(coeff, max=0) * this_slops[i][1]

        coeff, const = vmodel.weight_back(coeff, const, layer_id=i)
        # print(const)

    if coeff.dim() - 1 < len(vmodel.input_size):
        shape_change = [coeff.shape[0]]
        shape_change.extend(vmodel.input_size)
        coeff = torch.reshape(coeff, shape_change)

    const = const - torch.sum(torch.sum(coeff, dim=[2, 3]) * vmodel.mean / vmodel.std, dim=1)  # reduced dim: 2 ... last
    coeff = coeff / vmodel.std_l
    # pce for pesudo-counterexample
    xmin = torch.clip(x - eps, min=0)
    xmax = torch.clip(x + eps, max=1)
    val, pce = evaluate_inf_min_arg(xmin, xmax, coeff)
    # print(val, const)
    val = val + const
    # print(val)
    # print(pce.shape)
    # print(vmodel.check(pce, c))
    return val, pce


def verify_domain_beta(vmodel: VMODEL, domains: DOMAIN, x, eps, c, lr=0.04, decay=0.9):
    this_slops = deepcopy(vmodel.saved_slops)
    this_intercepts = deepcopy(vmodel.saved_intercepts)
    batch_size = domains.batch_size

    beta_neg = []
    beta_pos = []

    for i in range(vmodel.layer_num - 1):
        r_shape = [batch_size]
        r_shape.extend([1 for i in range(this_slops[i][0].dim())])
        this_slops[i] = list(this_slops[i])
        this_intercepts[i] = list(this_intercepts[i])
        this_slops[i][0] = this_slops[i][0].repeat(r_shape)
        this_slops[i][1] = this_slops[i][1].repeat(r_shape)
        this_intercepts[i][0] = this_intercepts[i][0].repeat(r_shape)
        this_intercepts[i][1] = this_intercepts[i][1].repeat(r_shape)
        # print(this_slops[i][0].shape)
        # print(this_intercepts[i][0].shape)

        if domains.pos_id[i] is None:
            beta_pos.append(torch.rand(0, requires_grad=True))
        else:
            this_slops[i][0][domains.pos_id[i]] = 1.0
            this_slops[i][1][domains.pos_id[i]] = 1.0
            this_intercepts[i][0][domains.pos_id[i]] = 0.0
            this_intercepts[i][1][domains.pos_id[i]] = 0.0
            num = domains.pos_id[i][0].numel()
            beta_pos.append(torch.rand(num, requires_grad=True, device=vmodel.device))
        if domains.neg_id[i] is None:
            beta_neg.append(torch.rand(0, requires_grad=True))
        else:
            this_slops[i][0][domains.neg_id[i]] = 0.0
            this_slops[i][1][domains.neg_id[i]] = 0.0
            this_intercepts[i][0][domains.neg_id[i]] = 0.0
            this_intercepts[i][1][domains.neg_id[i]] = 0.0
            num = domains.neg_id[i][0].numel()
            beta_neg.append(torch.rand(num, requires_grad=True, device=vmodel.device))

    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)
    c = c.repeat(batch_size, 1)

    maxval = None
    maxpce = None

    iter_time = 50
    for itt in range(iter_time):
        # print(itt)

        coeff, const = vmodel.weight_back(c, torch.tensor(0.0, device=vmodel.device), layer_id=len(vmodel.children) - 1)

        for i in range(vmodel.layer_num - 2, -1, -1):
            if coeff.dim() - 1 < len(vmodel.hidden_size[i]):
                shape_change = [coeff.shape[0]]
                shape_change.extend(vmodel.hidden_size[i])
                coeff = torch.reshape(coeff, shape_change)
            inc_const = torch.clip(coeff, min=0) * this_intercepts[i][0] + torch.clip(coeff, max=0) * \
                        this_intercepts[i][1]
            const += torch.sum(inc_const, dim=[d for d in range(1, inc_const.dim())])
            coeff = torch.clip(coeff, min=0) * this_slops[i][0] + torch.clip(coeff, max=0) * this_slops[i][1]

            if len(beta_pos[i]) > 0:
                beta_pos[i].requires_grad = True
                coeff[domains.pos_id[i]] -= beta_pos[i]
            if len(beta_neg[i]) > 0:
                beta_neg[i].requires_grad = True
                coeff[domains.neg_id[i]] += beta_neg[i]

            coeff, const = vmodel.weight_back(coeff, const, layer_id=i)

        if coeff.dim() - 1 < len(vmodel.input_size):
            shape_change = [coeff.shape[0]]
            shape_change.extend(vmodel.input_size)
            coeff = torch.reshape(coeff, shape_change)

        const = const - torch.sum(torch.sum(coeff, dim=[2, 3]) * vmodel.mean / vmodel.std,
                                  dim=1)  # reduced dim: 2 ... last
        coeff = coeff / vmodel.std_l
        # pce for pesudo-counterexample
        xmin = torch.clip(x - eps, min=0)
        xmax = torch.clip(x + eps, max=1)
        val, pce = evaluate_inf_min_arg(xmin, xmax, coeff)
        # print(val, const)
        ret = val + const

        loss = -1 * ret + 1.0
        for r in range(loss.numel() - 1):
            loss[r].backward(retain_graph=True)
        loss[-1].backward()
        # print(ret.detach())
        # print(beta_pos)
        # print(beta_neg)

        # loss = -1 * torch.mean(ret)
        # loss.backward()
        # if maxval is None:
        #     maxval = ret.clone().detach()
        #     maxpce = pce
        # else:
        #     less_id = torch.where(ret.detach() > maxval)
        #     if torch.where(ret.detach() != maxval)[0].numel() == 0:
        #         break
        #     maxval[less_id] = ret.detach()[less_id]
        #     maxpce = pce[less_id]

        for i in range(vmodel.layer_num - 1):
            if len(beta_pos[i]) > 0:
                beta_pos[i] = torch.clip(beta_pos[i] - lr*beta_pos[i].grad, min=0.0).detach_()
            if len(beta_neg[i]) > 0:
                beta_neg[i] = torch.clip(beta_neg[i] - lr*beta_neg[i].grad, min=0.0).detach_()
    maxval = ret.detach()
    maxpce = pce

    return maxval, maxpce


def verify_domain_beta_choice(vmodel: VMODEL, domains: DOMAIN, x, eps, c, lr=0.05, decay=0.99):
    this_slops = deepcopy(vmodel.saved_slops)
    this_intercepts = deepcopy(vmodel.saved_intercepts)
    this_bounds = deepcopy(vmodel.hidden_bounds)
    batch_size = domains.batch_size

    beta_neg = []
    beta_pos = []

    for i in range(vmodel.layer_num - 1):
        r_shape = [batch_size]
        r_shape.extend([1 for i in range(this_slops[i][0].dim())])
        this_slops[i] = list(this_slops[i])
        this_intercepts[i] = list(this_intercepts[i])
        this_bounds[i] = list(this_bounds[i])
        this_slops[i][0] = this_slops[i][0].repeat(r_shape)
        this_slops[i][1] = this_slops[i][1].repeat(r_shape)
        this_intercepts[i][0] = this_intercepts[i][0].repeat(r_shape)
        this_intercepts[i][1] = this_intercepts[i][1].repeat(r_shape)
        this_bounds[i][0] = this_bounds[i][0].repeat(r_shape)
        this_bounds[i][1] = this_bounds[i][1].repeat(r_shape)
        # print(this_slops[i][0].shape)
        # print(this_intercepts[i][0].shape)

        if domains.pos_id[i] is None:
            beta_pos.append(torch.rand(0, requires_grad=True, device=vmodel.device))
        else:
            this_slops[i][0][domains.pos_id[i]] = 1.0
            this_slops[i][1][domains.pos_id[i]] = 1.0
            this_intercepts[i][0][domains.pos_id[i]] = 0.0
            this_intercepts[i][1][domains.pos_id[i]] = 0.0
            this_bounds[i][0][domains.pos_id[i]] = 0.0
            num = domains.pos_id[i][0].numel()
            beta_pos.append(torch.rand(num, requires_grad=True, device=vmodel.device))
        if domains.neg_id[i] is None:
            beta_neg.append(torch.rand(0, requires_grad=True, device=vmodel.device))
        else:
            this_slops[i][0][domains.neg_id[i]] = 0.0
            this_slops[i][1][domains.neg_id[i]] = 0.0
            this_intercepts[i][0][domains.neg_id[i]] = 0.0
            this_intercepts[i][1][domains.neg_id[i]] = 0.0
            this_bounds[i][1][domains.neg_id[i]] = 0.0
            num = domains.neg_id[i][0].numel()
            beta_neg.append(torch.rand(num, requires_grad=True, device=vmodel.device))

    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)
    c = c.repeat(batch_size, 1)

    coeff_trace = []
    const_trace = []

    iter_time = 40
    for itt in range(iter_time):
        # print(itt)

        coeff, const = vmodel.weight_back(c, torch.tensor(0.0, device=vmodel.device), layer_id=len(vmodel.children) - 1)

        for i in range(vmodel.layer_num - 2, -1, -1):
            if coeff.dim() - 1 < len(vmodel.hidden_size[i]):
                shape_change = [coeff.shape[0]]
                shape_change.extend(vmodel.hidden_size[i])
                coeff = torch.reshape(coeff, shape_change)

            coeff_trace = [coeff.clone().detach()] + coeff_trace
            const_trace = [const.clone().detach()] + const_trace

            inc_const = torch.clip(coeff, min=0) * this_intercepts[i][0] + torch.clip(coeff, max=0) * \
                        this_intercepts[i][1]
            const += torch.sum(inc_const, dim=[d for d in range(1, inc_const.dim())])
            coeff = torch.clip(coeff, min=0) * this_slops[i][0] + torch.clip(coeff, max=0) * this_slops[i][1]

            if len(beta_pos[i]) > 0:
                beta_pos[i].requires_grad = True
                coeff[domains.pos_id[i]] -= beta_pos[i]
            if len(beta_neg[i]) > 0:
                beta_neg[i].requires_grad = True
                coeff[domains.neg_id[i]] += beta_neg[i]

            coeff, const = vmodel.weight_back(coeff, const, layer_id=i)

        if coeff.dim() - 1 < len(vmodel.input_size):
            shape_change = [coeff.shape[0]]
            shape_change.extend(vmodel.input_size)
            coeff = torch.reshape(coeff, shape_change)

        const = const - torch.sum(torch.sum(coeff, dim=[2, 3]) * vmodel.mean / vmodel.std,
                                  dim=1)  # reduced dim: 2 ... last
        coeff = coeff / vmodel.std_l
        # pce for pesudo-counterexample
        xmin = torch.clip(x - eps, min=0)
        xmax = torch.clip(x + eps, max=1)
        val, pce = evaluate_inf_min_arg(xmin, xmax, coeff)
        # print(val, const)
        ret = val + const
        # print(ret.detach())
        # print(beta_pos)
        # print(beta_neg)

        loss = -1 * ret + 1.0
        for r in range(loss.numel() - 1):
            loss[r].backward(retain_graph=True)
        loss[-1].backward()

        # if maxval is None:
        #     maxval = ret.clone().detach()
        #     maxpce = pce
        # else:
        #     less_id = torch.where(ret.detach() > maxval)
        #     if torch.where(ret.detach() != maxval)[0].numel() == 0:
        #         break
        #     maxval[less_id] = ret.detach()[less_id]
        #     maxpce = pce[less_id]

        if itt < iter_time - 1:
            for i in range(vmodel.layer_num - 1):
                if len(beta_pos[i]) > 0:
                    beta_pos[i] = torch.clip(beta_pos[i] - lr*beta_pos[i].grad, min=0.0).detach_()
                if len(beta_neg[i]) > 0:
                    beta_neg[i] = torch.clip(beta_neg[i] - lr*beta_neg[i].grad, min=0.0).detach_()

    for i in range(vmodel.layer_num - 1):
        beta_pos[i].requires_grad = False
        beta_neg[i].requires_grad = False
    maxval = ret.clone().detach()
    maxpce = pce

    pos = (pce - vmodel.mean_l) / vmodel.std_l

    best_score = None
    best_layer = None
    best_idx = None

    for i in range(vmodel.layer_num - 1):
        if pos.dim() - 1 > len(vmodel.hidden_size[i]):
            pos = torch.flatten(pos, start_dim=1)
        pre = vmodel.children[i](pos)

        und_bool = torch.logical_and(this_bounds[i][0] < 0, this_bounds[i][1] > 0)
        # und = torch.where(und_bool)
        slops = torch.where(coeff_trace[i] > 0, this_slops[i][0], this_slops[i][1])
        intercepts = torch.where(coeff_trace[i] > 0, this_intercepts[i][0], this_intercepts[i][1])
        pos = slops * pre + intercepts

        # incp = torch.maximum(coeff_trace[i], torch.zeros_like(coeff_trace[i])) * (
        #             (1 - this_slops[i][0]) * pre - this_intercepts[i][0]) + \
        #        torch.minimum(coeff_trace[i], torch.zeros_like(coeff_trace[i])) * (
        #                    (1 - this_slops[i][1]) * pre - this_intercepts[i][1])
        incn = -1 * pos * coeff_trace[i]
        # incn = torch.maximum(coeff_trace[i], torch.zeros_like(coeff_trace[i])) * (-1 * this_slops[i][0] * pre - this_intercepts[i][0]) + \
        #        torch.minimum(coeff_trace[i], torch.zeros_like(coeff_trace[i])) * (-1 * this_slops[i][1] * pre - this_intercepts[i][1])
        incp = incn + coeff_trace[i] * pre

        # compensation part start
        # compensation = torch.abs(coeff_trace[i] * pre)
        # pcom = torch.where(pre < 0)
        # ncom = torch.where(pre >= 0)
        # incp[pcom] += compensation[pcom]
        # incn[ncom] += compensation[ncom]
        #
        # incn = torch.minimum(incn, 0.2 - torch.ones_like(incn) * maxval.view([batch_size] + [1] * (incn.dim() - 1)))
        # incp = torch.minimum(incp, 0.2 - torch.ones_like(incp) * maxval.view([batch_size] + [1] * (incp.dim() - 1)))
        # compensation part end

        score = incp + incn
        # score = torch.minimum(incp, incn)

        nconcern = torch.where(torch.logical_not(und_bool))
        score[nconcern] = -10.0
        layer_score, max_id = torch.max(torch.flatten(score, start_dim=1), dim=1)
        real_id = [torch.arange(0, batch_size)] + recover_id(max_id, vmodel.hidden_size[i])
        # real_id = torch.where(score == layer_score.view([batch_size] + [1]*(coeff_trace[i].dim()-1)))
        # print(layer_score)
        # print(real_id)
        # print(score[und].shape)
        # actid = torch.where(this_bounds[i][0] > 0)
        # unactid = torch.where(this_bounds[i][1] < 0)

        # print(score[actid])
        # real_id = [und[t][mid] for t in range(len(und))]
        # print(this_bounds[i][0][real_id], this_bounds[i][1][real_id])
        # print(incp[real_id], incn[real_id])
        # print(score[real_id])

        if i == 0:
            best_score = layer_score
            best_layer = np.array([i] * batch_size)
            best_idx = [[real_id[l][k:k+1] for l in range(1, score.dim())] for k in range(batch_size)]
        else:
            better_idx = torch.where(layer_score > best_score)[0].tolist()
            best_score[better_idx] = layer_score[better_idx]
            best_layer[better_idx] = [i] * len(better_idx)
            for e in better_idx:
                best_idx[e] = [real_id[j][e:e+1] for j in range(1, score.dim())]
        # print('-'*40)

    best_layer = best_layer.tolist()
    # print(list(zip(best_layer, best_idx)))

    # print(best_idx)

    return maxval, maxpce, list(zip(best_layer, best_idx))


def pce_forward(vmodel: VMODEL, domains: DOMAIN, x, eps, c):
    this_slops = deepcopy(vmodel.saved_slops)
    this_intercepts = deepcopy(vmodel.saved_intercepts)
    batch_size = domains.batch_size

    for i in range(vmodel.layer_num - 1):
        r_shape = [batch_size]
        r_shape.extend([1 for i in range(this_slops[i][0].dim())])
        this_slops[i] = list(this_slops[i])
        this_intercepts[i] = list(this_intercepts[i])
        this_slops[i][0] = this_slops[i][0].repeat(r_shape)
        this_slops[i][1] = this_slops[i][1].repeat(r_shape)
        this_intercepts[i][0] = this_intercepts[i][0].repeat(r_shape)
        this_intercepts[i][1] = this_intercepts[i][1].repeat(r_shape)
        # print(this_slops[i][0].shape)
        # print(this_intercepts[i][0].shape)

        if not (domains.pos_id[i] is None):
            this_slops[i][0][domains.pos_id[i]] = 1.0
            this_slops[i][1][domains.pos_id[i]] = 1.0
            this_intercepts[i][0][domains.pos_id[i]] = 0.0
            this_intercepts[i][1][domains.pos_id[i]] = 0.0
        if not (domains.neg_id[i] is None):
            this_slops[i][0][domains.neg_id[i]] = 0.0
            this_slops[i][1][domains.neg_id[i]] = 0.0
            this_intercepts[i][0][domains.neg_id[i]] = 0.0
            this_intercepts[i][1][domains.neg_id[i]] = 0.0

    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)
    c = c.repeat(batch_size, 1)

    pce_trace = []
    coeff_trace = []
    const_trace = []
    coeff_trace1 = []
    const_trace1 = []

    coeff, const = vmodel.weight_back(c, b, layer_id=len(vmodel.children) - 1)

    for i in range(vmodel.layer_num - 2, -1, -1):
        if coeff.dim() - 1 < len(vmodel.hidden_size[i]):
            shape_change = [coeff.shape[0]]
            shape_change.extend(vmodel.hidden_size[i])
            coeff = torch.reshape(coeff, shape_change)

        coeff_trace1 = [coeff.clone()] + coeff_trace1
        const_trace1 = [const.clone()] + const_trace1
        pos_trace = torch.where(coeff > 0)
        neg_trace = torch.where(coeff < 0)
        pce_trace = [(neg_trace, pos_trace)] + pce_trace


        inc_const = torch.clip(coeff, min=0) * this_intercepts[i][0] + torch.clip(coeff, max=0) * \
                    this_intercepts[i][1]
        const += torch.sum(inc_const, dim=[d for d in range(1, inc_const.dim())])
        coeff = torch.clip(coeff, min=0) * this_slops[i][0] + torch.clip(coeff, max=0) * this_slops[i][1]

        coeff_trace = [coeff.clone()] + coeff_trace
        const_trace = [const.clone()] + const_trace

        coeff, const = vmodel.weight_back(coeff, const, layer_id=i)
        # coeff_trace = [coeff] + coeff_trace
        # const_trace = [const] + const_trace
        # print(const)

    const = const - torch.sum(torch.sum(coeff, dim=[2, 3]) * vmodel.mean / vmodel.std, dim=1)  # reduced dim: 2 ... last
    coeff = coeff / vmodel.std
    # pce for pesudo-counterexample
    xmin = torch.clip(x - eps, min=0)
    xmax = torch.clip(x + eps, max=1)
    val, pce = evaluate_inf_min_arg(xmin, xmax, coeff)
    # print(val, const)
    val = val + const

    pre_trace = []
    post_trace = []
    pce_normalize = (pce - vmodel.mean) / vmodel.std

    # print('0:', torch.sum(coeff_trace[0] * pce_normalize) + const_trace[0])

    pre = vmodel.children[0](pce_normalize)
    pos = torch.zeros_like(pre)
    pos[pce_trace[0][0]] = this_slops[0][1][pce_trace[0][0]] * pre[pce_trace[0][0]] + this_intercepts[0][1][pce_trace[0][0]]
    pos[pce_trace[0][1]] = this_slops[0][0][pce_trace[0][1]] * pre[pce_trace[0][1]] + this_intercepts[0][0][
        pce_trace[0][1]]
    # tkl = torch.where(coeff_trace[0] > 0)
    # tku = torch.where(coeff_trace[0] < 0)
    # pos[tkl] = this_slops[0][0][tkl] * pre[tkl] + this_intercepts[0][0][tkl]
    # pos[tku] = this_slops[0][1][tku] * pre[tku] + this_intercepts[0][1][tku]
    pre_trace.append(pre)
    post_trace.append(pos)
    # inc0 = coeff[pce_trace[0][0]] * (pos - )
    print('0:', torch.sum(coeff_trace[0] * pre) + const_trace[0])
    print('0:', torch.sum(coeff_trace1[0] * pos) + const_trace1[0])

    pre = vmodel.children[1](pos)
    pos = torch.zeros_like(pre)
    pos[pce_trace[1][0]] = this_slops[1][1][pce_trace[1][0]] * pre[pce_trace[1][0]] + this_intercepts[1][1][
        pce_trace[1][0]]
    pos[pce_trace[1][1]] = this_slops[1][0][pce_trace[1][1]] * pre[pce_trace[1][1]] + this_intercepts[1][0][
        pce_trace[1][1]]
    pre_trace.append(pre)
    post_trace.append(pos)
    print('1:', torch.sum(coeff_trace[1] * pre) + const_trace[1])
    print('1:', torch.sum(coeff_trace1[1] * pos) + const_trace1[1])

    pos = torch.flatten(pos, start_dim=1)
    pre = vmodel.children[2](pos)
    pos = torch.zeros_like(pre)
    pos[pce_trace[2][0]] = this_slops[2][1][pce_trace[2][0]] * pre[pce_trace[2][0]] + this_intercepts[2][1][
        pce_trace[2][0]]
    pos[pce_trace[2][1]] = this_slops[2][0][pce_trace[2][1]] * pre[pce_trace[2][1]] + this_intercepts[2][0][
        pce_trace[2][1]]
    pre_trace.append(pre)
    post_trace.append(pos)
    print('2:', torch.sum(coeff_trace[2] * pre) + const_trace[2])
    print('2:', torch.sum(coeff_trace1[2] * pos) + const_trace1[2])

    last_pre = vmodel.children[3](pos)

    print(pre_trace[2][0, 58])
    print(post_trace[2][0, 58])
    print(coeff_trace1[2][0, 58])
    # print(pos[0, 58])

    incp = coeff_trace1[2][0, 58] * (pre[0, 58] - this_slops[2][0][0, 58] * pre[0, 58] - this_intercepts[2][0][0, 58])
    incn = -1 * coeff_trace1[2][0, 58] * (this_slops[2][0][0, 58] * pre[0, 58] - this_intercepts[2][0][0, 58])
    print(incp, incn)
    print(incn + coeff_trace1[2][0, 58] * pos[0, 58])
    # print(this_slops[2][0][0, 58])
    # print(this_intercepts[2][0][0, 58])
    # print(vmodel.hidden_bounds[-1][0][58])
    # print(vmodel.hidden_bounds[-1][1][58])

    # choice = (2, [torch.tensor([58])])
    dlist = split_single(odom, choice)
    val1, _ = verify_domain_simple(vmodel, dlist[0], img, eps, c)
    val2, _ = verify_domain_simple(vmodel, dlist[1], img, eps, c)
    print(val1, val2, 'kkk')
    print(val + incp, val + incn)

    print(last_pre)
    print(torch.sum(last_pre * c))
    print(vmodel.model((pce - vmodel.mean) / vmodel.std))
    print(vmodel.check(pce, c))

    return val, pce


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
    eps = 0.15

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
    c = torch.zeros(10, dtype=torch.float)
    l = labels[i].item()
    c[l] = 1.0
    if l == 0:
        c[1] = -1.0
    else:
        c[0] = -1.0

    # c = torch.tensor([[[0, 1], [-1, -1]]])
    # d = c.unsqueeze(0)
    # a = torch.cat([d, d], dim=0)
    # # new_c = torch.tile(c, (2, 1))
    # # print(new_c.shape)
    # ida = torch.where(c > 0)
    # idb = torch.where(c < 0)
    # print(ida)
    # print(idb)
    # # new_id = [torch.cat([ida[i], idb[i]]) for i in range(len(ida))]
    # new_id = []
    # batch_id = []
    # batch_id.extend([0] * ida[0].shape[0])
    # batch_id.extend([1] * idb[0].shape[0])
    # new_id.append(torch.tensor(batch_id))
    # for i in range(len(ida)):
    #     new_id.append(torch.cat([ida[i], idb[i]]))
    # print(new_id)
    # print(a[new_id])
    # quit()
    r, opce = vmodel.backward_propagation(img, eps, c)
    print(r)
    print('*'*50)
    odom = DOMAIN(vmodel.layer_num - 1)

    idx = torch.where(torch.logical_and(vmodel.hidden_bounds[-1][0] < 0, vmodel.hidden_bounds[-1][1] > 0))
    print(idx)

    # force_id = [idx[i][7] for i in range(len(idx))]
    # n_id_pos = [torch.tensor([0]*1)]
    # n_id_pos.extend(force_id)
    # n_id_neg = [torch.tensor([0]*1)]
    # n_id_neg.extend([idx[i][7] for i in range(len(idx))])
    # dom.pos_id[-1] = n_id_pos
    # # dom.neg_id[-1] = n_id_neg
    # dom.batch_size = 1
    from domain import split_single
    choice = (2, [torch.tensor([58])])
    dlist = split_single(odom, choice)
    dom = dlist[1]
    print(dom)
    # quit()

    val, _ = verify_domain_beta(vmodel, dom, img, eps, c)
    print('='*20)
    val2, _ = pce_forward(vmodel, odom, img, eps, c)
    print('=' * 20)
    print(val, val2)
    print(val)
