import torch
import numpy as np
from copy import deepcopy
from bouding import verify_domain_simple, verify_domain_simple_wk, verify_domain_beta
from domain import split_single, make_domain_batch
from util import evaluate_inf_min_arg


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


def babsr_single(vmodel, domain, c, b=torch.tensor(0.0)):
    this_slops = deepcopy(vmodel.saved_slops)
    this_intercepts = deepcopy(vmodel.saved_intercepts)
    this_bounds = deepcopy(vmodel.hidden_bounds)
    batch_size = domain.batch_size

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

        if not (domain.pos_id[i] is None):
            this_slops[i][0][domain.pos_id[i]] = 1.0
            this_slops[i][1][domain.pos_id[i]] = 1.0
            this_intercepts[i][0][domain.pos_id[i]] = 0.0
            this_intercepts[i][1][domain.pos_id[i]] = 0.0
            this_bounds[i][0][domain.pos_id[i]] = 0.0
        if not (domain.neg_id[i] is None):
            this_slops[i][0][domain.neg_id[i]] = 0.0
            this_slops[i][1][domain.neg_id[i]] = 0.0
            this_intercepts[i][0][domain.neg_id[i]] = 0.0
            this_intercepts[i][1][domain.neg_id[i]] = 0.0
            this_bounds[i][1][domain.neg_id[i]] = 0.0

    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)
    c = -1 * c.repeat(batch_size, 1)

    best_score = None
    best_layer = None
    best_idx = None

    # print(domain)

    coeff, _ = vmodel.weight_back(c, b, layer_id=len(vmodel.children) - 1)

    for i in range(vmodel.layer_num - 2, -1, -1):
        if coeff.dim() - 1 < len(vmodel.hidden_size[i]):
            shape_change = [coeff.shape[0]]
            shape_change.extend(vmodel.hidden_size[i])
            coeff = torch.reshape(coeff, shape_change)

        # only for und neurons
        und = torch.where(torch.logical_and(this_bounds[i][0] < 0, this_bounds[i][1] > 0))
        negative = torch.where(this_bounds[i][1] < 0)
        # print(und)
        bias = vmodel.children[i].bias
        if coeff.dim() > 2:
            for j in range(coeff.dim() - bias.dim() - 1):
                bias = bias.unsqueeze(1)
        # print(bias.shape)
        # print(coeff.shape[1:])
        # print(und)
        k = this_bounds[i][1] / (this_bounds[i][1] - this_bounds[i][0])
        score = torch.maximum(coeff * bias, torch.zeros_like(coeff))\
                - k * coeff * bias + k * this_bounds[i][0] * torch.clip(coeff, min=0.0)
        score = torch.abs(score)
        idxes = torch.argmax(score[und])
        # print(score[und])
        # print(torch.max(score[und]))
        # print(idxes)
        real_id = [und[t][idxes] for t in range(len(und))]
        # print(real_id)
        # print(score[real_id])
        # print('='*20)

        if best_score is None:
            best_score = score[real_id]
            best_layer = i
            best_idx = real_id
        else:
            if score[real_id] > best_score:
                best_score = score[real_id]
                best_layer = i
                best_idx = real_id
        # print(best_score)
        # print(best_layer)
        # print(best_idx)

        # coeff[negative] = 0.0
        # coeff[und] = k[und] * coeff[und]

        coeff = torch.clip(coeff, min=0) * this_slops[i][0] + torch.clip(coeff, max=0) * this_slops[i][1]

        coeff, _ = vmodel.weight_back(coeff, b, layer_id=i)

    choice_idx = [best_idx[i].unsqueeze(0) for i in range(1, len(best_idx))]
    return (best_layer, choice_idx), best_score


def babsr_batch(vmodel, domain, c):
    this_slops = deepcopy(vmodel.saved_slops)
    this_intercepts = deepcopy(vmodel.saved_intercepts)
    this_bounds = deepcopy(vmodel.hidden_bounds)
    batch_size = domain.batch_size

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

        if not (domain.pos_id[i] is None):
            this_slops[i][0][domain.pos_id[i]] = 1.0
            this_slops[i][1][domain.pos_id[i]] = 1.0
            this_intercepts[i][0][domain.pos_id[i]] = 0.0
            this_intercepts[i][1][domain.pos_id[i]] = 0.0
            this_bounds[i][0][domain.pos_id[i]] = 0.0
        if not (domain.neg_id[i] is None):
            this_slops[i][0][domain.neg_id[i]] = 0.0
            this_slops[i][1][domain.neg_id[i]] = 0.0
            this_intercepts[i][0][domain.neg_id[i]] = 0.0
            this_intercepts[i][1][domain.neg_id[i]] = 0.0
            this_bounds[i][1][domain.neg_id[i]] = 0.0

    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)
    c = -1 * c.repeat(batch_size, 1)

    b = torch.tensor(0.0, device=vmodel.device)

    best_score = None
    best_layer = None
    best_idx = None

    # print(domain)

    coeff, _ = vmodel.weight_back(c, b, layer_id=len(vmodel.children) - 1)

    for i in range(vmodel.layer_num - 2, -1, -1):
        if coeff.dim() - 1 < len(vmodel.hidden_size[i]):
            shape_change = [coeff.shape[0]]
            shape_change.extend(vmodel.hidden_size[i])
            coeff = torch.reshape(coeff, shape_change)

        # only for und neurons
        und_bool = torch.logical_and(this_bounds[i][0] < 0, this_bounds[i][1] > 0)
        und = torch.where(und_bool)
        negative = torch.where(this_bounds[i][1] < 0)
        # print(und)
        bias = vmodel.children[i].bias
        if coeff.dim() > 2:
            for j in range(coeff.dim() - bias.dim() - 1):
                bias = bias.unsqueeze(1)
        # print(bias.shape)
        # print(coeff.shape[1:])
        # print(und)
        k = this_bounds[i][1] / (this_bounds[i][1] - this_bounds[i][0])
        score = torch.maximum(coeff * bias, torch.zeros_like(coeff))\
                - k * coeff * bias + k * this_bounds[i][0] * torch.clip(coeff, min=0.0)
        score = torch.abs(score)

        nconcern = torch.where(torch.logical_not(und_bool))
        score[nconcern] = -10.0
        layer_score, max_id = torch.max(torch.flatten(score, start_dim=1), dim=1)
        real_id = [torch.arange(0, batch_size)] + recover_id(max_id, vmodel.hidden_size[i])
        # real_id = torch.where(score == layer_score.view([batch_size] + [1] * (coeff.dim() - 1)))
        # idxes = torch.argmax(score[und])
        # print(score[und])
        # print(torch.max(score[und]))
        # print(idxes)
        # real_id = [und[t][idxes] for t in range(len(und))]
        # print(real_id)


        if best_score is None:
            best_score = score[real_id]
            best_layer = np.array([i] * batch_size)
            best_idx = [[real_id[l][k:k+1] for l in range(1, score.dim())] for k in range(batch_size)]
        else:
            better_idx = torch.where(layer_score > best_score)[0].tolist()
            # print(better_idx)
            best_score[better_idx] = layer_score[better_idx]
            best_layer[better_idx] = [i] * len(better_idx)
            for e in better_idx:
                best_idx[e] = [real_id[j][e:e + 1] for j in range(1, score.dim())]
        # print(best_score)
        # print(best_layer)
        # print(best_idx)

        coeff[negative] = 0.0
        coeff[und] = k[und] * coeff[und]
        # coeff = torch.clip(coeff, min=0) * this_slops[i][1] + torch.clip(coeff, max=0) * this_slops[i][0]

        coeff, _ = vmodel.weight_back(coeff, b, layer_id=i)

    # choice_idx = [best_idx[i].unsqueeze(0) for i in range(1, len(best_idx))]
    return list(zip(best_layer, best_idx)), best_score


def fsb_single(vmodel, domain, x, eps, c, b=torch.tensor(0.0)):
    this_slops = deepcopy(vmodel.saved_slops)
    this_intercepts = deepcopy(vmodel.saved_intercepts)
    this_bounds = deepcopy(vmodel.hidden_bounds)
    batch_size = domain.batch_size

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

        if not (domain.pos_id[i] is None):
            this_slops[i][0][domain.pos_id[i]] = 1.0
            this_slops[i][1][domain.pos_id[i]] = 1.0
            this_intercepts[i][0][domain.pos_id[i]] = 0.0
            this_intercepts[i][1][domain.pos_id[i]] = 0.0
            this_bounds[i][0][domain.pos_id[i]] = 0.0
        if not (domain.neg_id[i] is None):
            this_slops[i][0][domain.neg_id[i]] = 0.0
            this_slops[i][1][domain.neg_id[i]] = 0.0
            this_intercepts[i][0][domain.neg_id[i]] = 0.0
            this_intercepts[i][1][domain.neg_id[i]] = 0.0
            this_bounds[i][1][domain.neg_id[i]] = 0.0

    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)
    c = -1 * c.repeat(batch_size, 1)
    b = b.to(vmodel.device)

    best_score = None
    best_layer = None
    best_idx = None

    # print(domain)

    coeff, _ = vmodel.weight_back(c, b, layer_id=len(vmodel.children) - 1)

    for i in range(vmodel.layer_num - 2, -1, -1):
        if coeff.dim() - 1 < len(vmodel.hidden_size[i]):
            shape_change = [coeff.shape[0]]
            shape_change.extend(vmodel.hidden_size[i])
            coeff = torch.reshape(coeff, shape_change)

        # only for und neurons
        und = torch.where(torch.logical_and(this_bounds[i][0] < 0, this_bounds[i][1] > 0))
        negative = torch.where(this_bounds[i][1] < 0)
        # print(und)
        bias = vmodel.children[i].bias
        if coeff.dim() > 2:
            for j in range(coeff.dim() - bias.dim() - 1):
                bias = bias.unsqueeze(1)
        # print(bias.shape)
        # print(coeff.shape[1:])
        # print(und)
        k = this_bounds[i][1] / (this_bounds[i][1] - this_bounds[i][0])
        score = torch.minimum(coeff * bias, torch.zeros_like(coeff))\
                - k * coeff * bias + k * this_bounds[i][0] * torch.clip(coeff, min=0.0)
        score = torch.abs(score)
        idxes = torch.argmax(score[und])
        real_id = [und[t][idxes] for t in range(len(und))]
        candi = (i, [real_id[t].unsqueeze(0) for t in range(1, len(real_id))])
        dlist = split_single(domain, candi)
        dbatch = make_domain_batch(dlist)
        candi_vals, _ = verify_domain_simple(vmodel, dbatch, x, eps, -1 * c)
        candi_vals_wk, _ = verify_domain_simple_wk(vmodel, dbatch, x, eps, -1 * c)
        candi_score, _ = torch.max(torch.cat([candi_vals.unsqueeze(0), candi_vals_wk.unsqueeze(0)]), dim=0)
        candi_score = torch.min(candi_score)

        t_score = -1 * k * this_bounds[i][0] * torch.clip(coeff, min=0.0)
        t_idxes = torch.argmax(t_score[und])
        t_real_id = [und[t][t_idxes] for t in range(len(und))]
        t_candi = (i, [t_real_id[t].unsqueeze(0) for t in range(1, len(t_real_id))])
        t_dlist = split_single(domain, t_candi)
        t_dbatch = make_domain_batch(t_dlist)
        t_candi_vals, _ = verify_domain_simple(vmodel, t_dbatch, x, eps, c)
        t_candi_vals_wk, _ = verify_domain_simple_wk(vmodel, t_dbatch, x, eps, c)
        t_candi_score, _ = torch.max(torch.cat([t_candi_vals.unsqueeze(0), t_candi_vals_wk.unsqueeze(0)]), dim=0)
        t_candi_score = torch.min(t_candi_score)

        if t_candi_score > candi_score:
            layer_score = t_candi_score
            layer_idx = t_real_id
        else:
            layer_score = candi_score
            layer_idx = real_id

        if best_score is None:
            best_score = layer_score
            best_layer = i
            best_idx = layer_idx
        else:
            best_score = layer_score
            best_layer = i
            best_idx = layer_idx


        # inc_const = torch.clip(coeff, min=0) * this_intercepts[i][0] + torch.clip(coeff, max=0) * \
        #             this_intercepts[i][1]
        # const += torch.sum(inc_const, dim=[d for d in range(1, inc_const.dim())])
        coeff[negative] = 0.0
        coeff[und] = coeff[und] * k[und]

        coeff, _ = vmodel.weight_back(coeff, b, layer_id=i)

    choice_idx = [best_idx[i].unsqueeze(0) for i in range(1, len(best_idx))]
    return (best_layer, choice_idx), best_score


def fsb_batch(vmodel, domain, dom_list, x, eps, c):
    this_slops = deepcopy(vmodel.saved_slops)
    this_intercepts = deepcopy(vmodel.saved_intercepts)
    this_bounds = deepcopy(vmodel.hidden_bounds)
    batch_size = domain.batch_size

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

        if not (domain.pos_id[i] is None):
            this_slops[i][0][domain.pos_id[i]] = 1.0
            this_slops[i][1][domain.pos_id[i]] = 1.0
            this_intercepts[i][0][domain.pos_id[i]] = 0.0
            this_intercepts[i][1][domain.pos_id[i]] = 0.0
            this_bounds[i][0][domain.pos_id[i]] = 0.0
        if not (domain.neg_id[i] is None):
            this_slops[i][0][domain.neg_id[i]] = 0.0
            this_slops[i][1][domain.neg_id[i]] = 0.0
            this_intercepts[i][0][domain.neg_id[i]] = 0.0
            this_intercepts[i][1][domain.neg_id[i]] = 0.0
            this_bounds[i][1][domain.neg_id[i]] = 0.0

    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)
    c_o = -1 * c.repeat(batch_size, 1)
    b = torch.tensor(0.0, device=vmodel.device)

    best_score = None
    best_layer = None
    best_idx = None

    # print(domain)

    coeff, _ = vmodel.weight_back(c_o, b, layer_id=len(vmodel.children) - 1)

    for i in range(vmodel.layer_num - 2, -1, -1):
        if coeff.dim() - 1 < len(vmodel.hidden_size[i]):
            shape_change = [coeff.shape[0]]
            shape_change.extend(vmodel.hidden_size[i])
            coeff = torch.reshape(coeff, shape_change)

        # only for und neurons
        und_bool = torch.logical_and(this_bounds[i][0] < 0, this_bounds[i][1] > 0)
        und = torch.where(und_bool)
        negative = torch.where(this_bounds[i][1] < 0)
        # print(und)
        bias = vmodel.children[i].bias
        if coeff.dim() > 2:
            for j in range(coeff.dim() - bias.dim() - 1):
                bias = bias.unsqueeze(1)
        # print(bias.shape)
        # print(coeff.shape[1:])
        # print(und)
        k = this_bounds[i][1] / (this_bounds[i][1] - this_bounds[i][0])
        score = torch.minimum(coeff * bias, torch.zeros_like(coeff))\
                - k * coeff * bias + k * this_bounds[i][0] * torch.clip(coeff, min=0.0)
        score = torch.abs(score)
        t_score = -1 * k * this_bounds[i][0] * torch.clip(coeff, min=0.0)
        # t_score = torch.abs(t_score)
        nconcern = torch.where(torch.logical_not(und_bool))
        score[nconcern] = -10.0
        max_score, max_id = torch.max(torch.flatten(score, start_dim=1), dim=1)
        real_id = [torch.arange(0, batch_size)] + recover_id(max_id, vmodel.hidden_size[i])
        # real_id = torch.where(score == max_score.view([batch_size] + [1] * (coeff.dim() - 1)))
        # idxes = torch.argmax(score[und])
        t_score[nconcern] = -10.0
        t_max_score, t_max_id = torch.max(torch.flatten(t_score, start_dim=1), dim=1)
        t_real_id = [torch.arange(0, batch_size)] + recover_id(t_max_id, vmodel.hidden_size[i])
        # t_real_id = torch.where(t_score == t_max_score.view([batch_size] + [1] * (coeff.dim() - 1)))

        p_candi_dom = []
        n_candi_dom = []
        for j in range(batch_size):
            dp, dn = split_single(dom_list[j], (i, [real_id[k][j:j+1] for k in range(1, coeff.dim())]))
            p_candi_dom.append(dp)
            n_candi_dom.append(dn)
        for j in range(batch_size):
            t_dp, t_dn = split_single(dom_list[j], (i, [t_real_id[k][j:j+1] for k in range(1, coeff.dim())]))
            p_candi_dom.append(t_dp)
            n_candi_dom.append(t_dn)
        pdom = make_domain_batch(p_candi_dom)
        ndom = make_domain_batch(n_candi_dom)

        p_cr, _ = verify_domain_simple(vmodel, pdom, x, eps, c)
        p_wk, _ = verify_domain_simple_wk(vmodel, pdom, x, eps, c)
        n_cr, _ = verify_domain_simple(vmodel, ndom, x, eps, c)
        n_wk, _ = verify_domain_simple_wk(vmodel, ndom, x, eps, c)
        mix_score = torch.minimum(torch.maximum(p_cr, p_wk), torch.maximum(n_cr, n_wk))
        mix_score = torch.reshape(mix_score, (2, batch_size))
        layer_score, t_better = torch.max(mix_score, dim=0)
        # print(mix_score)
        # print(layer_score, t_better)
        layer_idx = []
        for j in range(batch_size):
            if t_better[j].item() == 0:
                layer_idx.append([real_id[k][j:j+1] for k in range(1, coeff.dim())])
            else:
                layer_idx.append([t_real_id[k][j:j + 1] for k in range(1, coeff.dim())])
        # print(layer_idx)

        if best_score is None:
            best_score = score[real_id]
            best_layer = np.array([i] * batch_size)
            best_idx = layer_idx
        else:
            better_idx = torch.where(layer_score > best_score)[0].tolist()
            # print(better_idx)
            best_score[better_idx] = layer_score[better_idx]
            best_layer[better_idx] = [i] * len(better_idx)
            for e in better_idx:
                best_idx[e] = [real_id[j][e:e + 1] for j in range(1, score.dim())]
        # print(best_score)
        # print(best_layer)
        # print(best_idx)

        # inc_const = torch.clip(coeff, min=0) * this_intercepts[i][0] + torch.clip(coeff, max=0) * \
        #             this_intercepts[i][1]
        # const += torch.sum(inc_const, dim=[d for d in range(1, inc_const.dim())])
        coeff[negative] = 0.0
        coeff[und] = coeff[und] * k[und]

        coeff, _ = vmodel.weight_back(coeff, b, layer_id=i)

    return list(zip(best_layer, best_idx)), best_score


def optimal_lift(vmodel, domain, x, eps, c, b=torch.tensor(0.0)):
    this_slops = deepcopy(vmodel.saved_slops)
    this_intercepts = deepcopy(vmodel.saved_intercepts)
    this_bounds = deepcopy(vmodel.hidden_bounds)
    batch_size = domain.batch_size

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

        if not (domain.pos_id[i] is None):
            this_slops[i][0][domain.pos_id[i]] = 1.0
            this_slops[i][1][domain.pos_id[i]] = 1.0
            this_intercepts[i][0][domain.pos_id[i]] = 0.0
            this_intercepts[i][1][domain.pos_id[i]] = 0.0
            this_bounds[i][0][domain.pos_id[i]] = 0.0
        if not (domain.neg_id[i] is None):
            this_slops[i][0][domain.neg_id[i]] = 0.0
            this_slops[i][1][domain.neg_id[i]] = 0.0
            this_intercepts[i][0][domain.neg_id[i]] = 0.0
            this_intercepts[i][1][domain.neg_id[i]] = 0.0
            this_bounds[i][1][domain.neg_id[i]] = 0.0

    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)
    c = c.repeat(batch_size, 1)
    b = b.to(vmodel.device)

    coeff_trace = []
    # const_trace = []

    coeff, _ = vmodel.weight_back(c, b, layer_id=len(vmodel.children) - 1)

    for i in range(vmodel.layer_num - 2, -1, -1):
        if coeff.dim() - 1 < len(vmodel.hidden_size[i]):
            shape_change = [coeff.shape[0]]
            shape_change.extend(vmodel.hidden_size[i])
            coeff = torch.reshape(coeff, shape_change)

        coeff_trace = [coeff.clone()] + coeff_trace
        # const_trace = [const.clone()] + const_trace

        # inc_const = torch.clip(coeff, min=0) * this_intercepts[i][0] + torch.clip(coeff, max=0) * \
        #             this_intercepts[i][1]
        # const += torch.sum(inc_const, dim=[d for d in range(1, inc_const.dim())])

        coeff = torch.clip(coeff, min=0) * this_slops[i][0] + torch.clip(coeff, max=0) * this_slops[i][1]

        # und = torch.where(torch.logical_and(this_bounds[i][0] < 0, this_bounds[i][1] > 0))
        # negative = torch.where(this_bounds[i][1] < 0)
        # coeff[negative] = 0.0
        # coeff[und] = coeff[und] * this_bounds[i][1][und] / (this_bounds[i][1][und] - this_bounds[i][0][und])

        coeff, _ = vmodel.weight_back(coeff, torch.tensor(0.0), layer_id=i)

    if coeff.dim() - 1 < len(vmodel.input_size):
        shape_change = [coeff.shape[0]]
        shape_change.extend(vmodel.input_size)
        coeff = torch.reshape(coeff, shape_change)
        
    # const = const - torch.sum(torch.sum(coeff, dim=[2, 3]) * vmodel.mean / vmodel.std, dim=1)  # reduced dim: 2 ... last
    coeff = coeff / vmodel.std_l
    # pce for pesudo-counterexample
    xmin = torch.clip(x - eps, min=0)
    xmax = torch.clip(x + eps, max=1)
    val, pce = evaluate_inf_min_arg(xmin, xmax, coeff)

    best_score = None
    best_layer = None
    best_idx = None

    pos = (pce - vmodel.mean_l) / vmodel.std_l
    for i in range(vmodel.layer_num - 1):
        if coeff.dim() - 1 > len(vmodel.hidden_size[i]):
            pos = torch.flatten(pos, start_dim=1)
        pre = vmodel.children[i](pos)
        und_bool = torch.logical_and(this_bounds[i][0] < 0, this_bounds[i][1] > 0)
        und = torch.where(und_bool)
        # p_id = torch.where(torch.logical_and(coeff_trace[i] > 0, und_bool))
        # n_id = torch.where(torch.logical_and(coeff_trace[i] < 0, und_bool))
        # pos = torch.zeros_like(pre)
        # pos[p_id] = this_slops[i][0][p_id] * pre[p_id] + this_intercepts[i][0][p_id]
        # pos[n_id] = this_slops[i][1][n_id] * pre[n_id] + this_intercepts[i][1][n_id]

        slops = torch.where(coeff_trace[i] > 0, this_slops[i][0], this_slops[i][1])
        intercepts = torch.where(coeff_trace[i] > 0, this_intercepts[i][0], this_intercepts[i][1])
        pos = slops * pre + intercepts

        # k = this_bounds[i][1] / (this_bounds[i][1] - this_bounds[i][0])

        incp = torch.maximum(coeff_trace[i], torch.zeros_like(coeff_trace[i])) * ((1 - this_slops[i][0]) * pre - this_intercepts[i][0]) + \
               torch.minimum(coeff_trace[i], torch.zeros_like(coeff_trace[i])) * ((1 - this_slops[i][1]) * pre - this_intercepts[i][1])
        incn = torch.maximum(coeff_trace[i], torch.zeros_like(coeff_trace[i])) * (-1 * this_slops[i][0] * pre - this_intercepts[i][0]) + \
               torch.minimum(coeff_trace[i], torch.zeros_like(coeff_trace[i])) * (-1 * this_slops[i][1] * pre - this_intercepts[i][1])

        # incp = torch.maximum(coeff_trace[i], torch.zeros_like(coeff_trace[i])) * ((1 - k) * pre) + \
        #        torch.minimum(coeff_trace[i], torch.zeros_like(coeff_trace[i])) * ((1 - k) * pre + this_bounds[i][0] * k)
        # incn = torch.maximum(coeff_trace[i], torch.zeros_like(coeff_trace[i])) * (-1 * k * pre) + \
        #        torch.minimum(coeff_trace[i], torch.zeros_like(coeff_trace[i])) * (-1 * k * pre + this_bounds[i][0] * k)

        score = incp + incn
        mid = torch.argmax(score[und])
        real_id = [und[t][mid] for t in range(len(und))]
        # print(score[real_id])
        # print(real_id)
        # print(incp[real_id], incn[real_id])
        # print(coeff_trace[i][real_id])
        # print(pre[real_id])
        # print(this_slops[i][0][real_id], this_intercepts[i][0][real_id])
        # print(this_slops[i][1][real_id], this_intercepts[i][1][real_id])

        # pos = pre.clone()
        # und = torch.where(torch.logical_and(this_bounds[i][0] < 0, this_bounds[i][1] > 0))
        # negative = torch.where(this_bounds[i][1] < 0)
        # pos[negative] = 0.0
        # intercepts = torch.where(coeff_trace[i] > 0, torch.zeros_like(pre), -1 * this_bounds[i][0] * k)
        # pos[und] = pre[und] * k[und] + intercepts[und]

        if best_score is None:
            best_score = score[real_id]
            best_layer = i
            best_idx = real_id
        else:
            if score[real_id] > best_score:
                best_score = score[real_id]
                best_layer = i
                best_idx = real_id

        # print(incp[real_id], incn[real_id])
        # print(real_id)
        # print(torch.sum(coeff_trace[i] * pos) + const_trace[i])
        #
        # choice = (i, [real_id[t].unsqueeze(0) for t in range(1, len(real_id))])
        # print(choice)
        # dlist = split_single(domain, choice)
        # val1, _ = verify_domain_simple(vmodel, dlist[0], x, eps, c)
        # val2, _ = verify_domain_simple(vmodel, dlist[1], x, eps, c)
        # print(val1, val2)
        # print('-'*40)

    choice_idx = [best_idx[i].unsqueeze(0) for i in range(1, len(best_idx))]
    return (best_layer, choice_idx), best_score


def rc_single(vmodel, domain, c, b=torch.tensor(0.0)):
    this_slops = deepcopy(vmodel.saved_slops)
    this_intercepts = deepcopy(vmodel.saved_intercepts)
    this_bounds = deepcopy(vmodel.hidden_bounds)
    batch_size = domain.batch_size

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

        if not (domain.pos_id[i] is None):
            this_slops[i][0][domain.pos_id[i]] = 1.0
            this_slops[i][1][domain.pos_id[i]] = 1.0
            this_intercepts[i][0][domain.pos_id[i]] = 0.0
            this_intercepts[i][1][domain.pos_id[i]] = 0.0
            this_bounds[i][0][domain.pos_id[i]] = 0.0
        if not (domain.neg_id[i] is None):
            this_slops[i][0][domain.neg_id[i]] = 0.0
            this_slops[i][1][domain.neg_id[i]] = 0.0
            this_intercepts[i][0][domain.neg_id[i]] = 0.0
            this_intercepts[i][1][domain.neg_id[i]] = 0.0
            this_bounds[i][1][domain.neg_id[i]] = 0.0

    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)
    c = -1 * c.repeat(batch_size, 1)

    best_score = None
    best_layer = None
    best_idx = None

    # print(domain)
    target_layer = torch.randint(low=0, high=vmodel.layer_num-1, size=[1]).item()
    real_id = torch.tensor([0])

    coeff, _ = vmodel.weight_back(c, b, layer_id=len(vmodel.children) - 1)

    for i in range(vmodel.layer_num - 2, -1, -1):
        if coeff.dim() - 1 < len(vmodel.hidden_size[i]):
            shape_change = [coeff.shape[0]]
            shape_change.extend(vmodel.hidden_size[i])
            coeff = torch.reshape(coeff, shape_change)

        # only for und neurons
        und = torch.where(torch.logical_and(this_bounds[i][0] < 0, this_bounds[i][1] > 0))
        negative = torch.where(this_bounds[i][1] < 0)
        # print(und)
        bias = vmodel.children[i].bias
        if coeff.dim() > 2:
            for j in range(coeff.dim() - bias.dim() - 1):
                bias = bias.unsqueeze(1)
        # print(bias.shape)
        # print(coeff.shape[1:])
        # print(und)
        k = this_bounds[i][1] / (this_bounds[i][1] - this_bounds[i][0])
        score = torch.maximum(coeff * bias, torch.zeros_like(coeff))\
                - k * coeff * bias + k * this_bounds[i][0] * torch.clip(coeff, min=0.0)
        # score = torch.abs(score)
        # idxes = torch.argmax(score[und])
        # print(score[und])
        # print(torch.max(score[und]))
        # print(idxes)
        # real_id = [und[t][idxes] for t in range(len(und))]

        if i == target_layer:
            real_id = torch.randint(low=0, high=int(score.numel()/batch_size), size=[batch_size])
            real_id = recover_id(real_id, vmodel.hidden_size[i])
        coeff[negative] = 0.0
        coeff[und] = k[und] * coeff[und]

        coeff, _ = vmodel.weight_back(coeff, b, layer_id=i)

    # choice_idx = [best_idx[i].unsqueeze(0) for i in range(1, len(best_idx))]
    return [target_layer, real_id], 0


def rc_batch(vmodel, domain, x, eps ,c, b=torch.tensor(0.0)):
    this_slops = deepcopy(vmodel.saved_slops)
    this_intercepts = deepcopy(vmodel.saved_intercepts)
    this_bounds = deepcopy(vmodel.hidden_bounds)
    batch_size = domain.batch_size

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

        if not (domain.pos_id[i] is None):
            this_slops[i][0][domain.pos_id[i]] = 1.0
            this_slops[i][1][domain.pos_id[i]] = 1.0
            this_intercepts[i][0][domain.pos_id[i]] = 0.0
            this_intercepts[i][1][domain.pos_id[i]] = 0.0
            this_bounds[i][0][domain.pos_id[i]] = 0.0
        if not (domain.neg_id[i] is None):
            this_slops[i][0][domain.neg_id[i]] = 0.0
            this_slops[i][1][domain.neg_id[i]] = 0.0
            this_intercepts[i][0][domain.neg_id[i]] = 0.0
            this_intercepts[i][1][domain.neg_id[i]] = 0.0
            this_bounds[i][1][domain.neg_id[i]] = 0.0

    if c.dim() == 1:
        c = torch.unsqueeze(c, 0)
    c = -1 * c.repeat(batch_size, 1)

    best_score = None
    best_layer = None
    best_idx = None
    best_pair = None

    # print(domain)
    target_layer = torch.randint(low=0, high=vmodel.layer_num - 1, size=[batch_size])
    real_id = [None] * batch_size

    coeff, _ = vmodel.weight_back(c, b, layer_id=len(vmodel.children) - 1)

    for i in range(vmodel.layer_num - 2, -1, -1):
        if coeff.dim() - 1 < len(vmodel.hidden_size[i]):
            shape_change = [coeff.shape[0]]
            shape_change.extend(vmodel.hidden_size[i])
            coeff = torch.reshape(coeff, shape_change)

        # only for und neurons
        und_bool = torch.logical_and(this_bounds[i][0] < 0, this_bounds[i][1] > 0)
        und = torch.where(und_bool)
        negative = torch.where(this_bounds[i][1] < 0)
        # print(und)
        bias = vmodel.children[i].bias
        if coeff.dim() > 2:
            for j in range(coeff.dim() - bias.dim() - 1):
                bias = bias.unsqueeze(1)
        # print(bias.shape)
        # print(coeff.shape[1:])
        # print(und)
        k = this_bounds[i][1] / (this_bounds[i][1] - this_bounds[i][0])
        incn = - k * coeff * bias + k * this_bounds[i][0] * torch.maximum(coeff, torch.zeros_like(coeff))
        incp = incn + coeff * bias
        # score = torch.maximum(coeff * bias, torch.zeros_like(coeff))\
        #         - k * coeff * bias + k * this_bounds[i][0] * torch.clip(coeff, min=0.0)
        score = torch.abs(torch.maximum(incn, incp))
        # score = torch.maximum(incn, incp)

        # nconcern = torch.where(torch.logical_not(und_bool))
        # score[nconcern] = -10.0

        # layer_score, max_id = torch.max(torch.flatten(score, start_dim=1), dim=1)
        # real_id = [torch.arange(0, batch_size)] + recover_id(max_id, vmodel.hidden_size[i])

        layer_id = torch.randint(low=0, high=int(score.numel() / batch_size), size=[batch_size])

        fix_id = torch.where(target_layer == i)[0]
        if len(fix_id) > 0:
            for eid in fix_id.tolist():
                real_id[eid] = recover_id(layer_id[eid:eid+1], vmodel.hidden_size[i])
        # real_id = torch.where(score == layer_score.view([batch_size] + [1] * (coeff.dim() - 1)))
        # idxes = torch.argmax(score[und])
        # print(score[und])
        # print(torch.max(score[und]))
        # print(idxes)
        # real_id = [und[t][idxes] for t in range(len(und))]
        # print(real_id)

        # oval, _ = vmodel.backward_propagation(x, eps, -1 * c)
        # from bouding import verify_domain_beta
        # import matplotlib.pyplot as plt


        # if best_score is None:
        #     best_score = score[real_id]
        #     best_layer = np.array([i] * batch_size)
        #     best_idx = [[real_id[l][k:k+1] for l in range(1, score.dim())] for k in range(batch_size)]
        #     best_pair = [incn[real_id], incp[real_id]]
        # else:
        #     better_idx = torch.where(layer_score > best_score)[0].tolist()
        #     # print(better_idx)
        #     best_score[better_idx] = layer_score[better_idx]
        #     best_layer[better_idx] = [i] * len(better_idx)
        #     for e in better_idx:
        #         best_idx[e] = [real_id[j][e:e + 1] for j in range(1, score.dim())]
        #     best_pair[0][better_idx] = incn[real_id][better_idx]
        #     best_pair[1][better_idx] = incp[real_id][better_idx]


        coeff[negative] = 0.0
        coeff[und] = k[und] * coeff[und]

        coeff, _ = vmodel.weight_back(coeff, b, layer_id=i)
    target_layer = target_layer.tolist()

    # choice_idx = [best_idx[i].unsqueeze(0) for i in range(1, len(best_idx))]
    return list(zip(target_layer, real_id)), 0
