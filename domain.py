import torch
from copy import deepcopy
from itertools import product
# from model_verification import VModel


class DOMAIN:
    def __init__(self, hidden_layer_num):
        self.hlayer_num  = hidden_layer_num
        self.batch_size = 1
        self.pos_id = [None] * hidden_layer_num
        self.neg_id = [None] * hidden_layer_num
        # self.additional_bounds = [None] * hidden_layer_num
        # self.acting = [False] * hidden_layer_num

    def add_constraint_single(self, layer, idxes, pos):
        # self.batch_size += 1
        d_idxes = [torch.tensor([0])] + idxes
        if pos:
            if self.pos_id[layer] is None:
                self.pos_id[layer] = d_idxes
            else:
                # print(d_idxes)
                # print(self.pos_id[layer])
                # torch.cat([self.pos_id[layer][1], d_idxes[1]])
                # print(1)
                self.pos_id[layer] = [torch.cat([self.pos_id[layer][i], d_idxes[i]]) for i in range(len(d_idxes))]
        else:
            if self.neg_id[layer] is None:
                self.neg_id[layer] = d_idxes
            else:
                self.neg_id[layer] = [torch.cat([self.neg_id[layer][i], d_idxes[i]]) for i in range(len(d_idxes))]

    def assign_batch_id(self, bid):
        for i in range(self.hlayer_num):
            if self.pos_id[i] is not None:
                self.pos_id[i][0] += bid
            if self.neg_id[i] is not None:
                self.neg_id[i][0] += bid

    def __repr__(self):
        show = ''
        show = show + '-'*20 + '\n'
        show = show + 'batch size: ' + str(self.batch_size) + '\n'
        show = show + 'pos id: ' + str(self.pos_id) + '\n'
        show = show + 'neg id: ' + str(self.neg_id) + '\n'
        show = show + '-'*20 + '\n'
        return show


class LIST:
    def __init__(self, l):
        self.content = list(l)

    def __len__(self):
        return len(self.content)

    def get(self):
        return self.content.pop(0)

    def push(self, item):
        self.content.append(item)

    def get_max(self):
        vals = [e[0] for e in self.content]
        return max(vals)


def split_single(dom: DOMAIN, decision):
    # dom's batch_size is 1, decision only contain 1 choice
    layer, indexes = decision
    dom1 = deepcopy(dom)
    dom2 = deepcopy(dom)
    dom1.add_constraint_single(layer, indexes, pos=True)
    dom2.add_constraint_single(layer, indexes, pos=False)

    return [dom1, dom2]


def make_domain_batch(dom_list):
    # the batch_size of each domain is 1
    hlayer_num = len(dom_list[0].pos_id)

    batch_dom = DOMAIN(hlayer_num)
    batch_dom.batch_size = len(dom_list)

    # for i in range(1, len(dom_list)):
    #     dom_list[i].assign_batch_id(i)

    for i in range(hlayer_num):
        for j in range(len(dom_list)):
            if dom_list[j].pos_id[i] is not None:
                batch_dom.pos_id[i] = cat_idxs(batch_dom.pos_id[i], dom_list[j].pos_id[i], j)
            if dom_list[j].neg_id[i] is not None:
                batch_dom.neg_id[i] = cat_idxs(batch_dom.neg_id[i], dom_list[j].neg_id[i], j)

        # temp_pos = [d.pos_id[i] for d in dom_list if d.pos_id[i] is not None]
        # if len(temp_pos) == 1:
        #     batch_dom.pos_id[i] = temp_pos[0]
        # if len(temp_pos) > 1:
        #     batch_dom.pos_id[i] = []
        #     for j in range(len(temp_pos[0])):
        #         batch_dom.pos_id[i].append(torch.cat([e[j] for e in temp_pos]))
        #
        # temp_neg = [d.neg_id[i] for d in dom_list if d.neg_id[i] is not None]
        # if len(temp_neg) == 1:
        #     batch_dom.neg_id[i] = temp_neg[0]
        # if len(temp_neg) > 1:
        #     batch_dom.neg_id[i] = []
        #     for j in range(len(temp_neg[0])):
        #         batch_dom.pos_id[i].append(torch.cat([e[j] for e in temp_neg]))

    return batch_dom


def cat_idxs(ori_id, new_id, batch_num):
    if ori_id is None:
        return [new_id[0] + batch_num] + new_id[1:]
    else:
        head = [torch.cat([ori_id[0], new_id[0] + batch_num])]
        return head + [torch.cat([ori_id[i], new_id[i]]) for i in range(1, len(ori_id))]


if __name__ == '__main__':

    oridom = DOMAIN(3)
    d = (2, [torch.tensor([1])])
    doms = split_single(oridom, d)

    oridom2 = DOMAIN(3)
    d = (2, [torch.tensor([2])])
    doms2 = split_single(oridom2, d)

    # a = [None, (0, 1), None, (0, 2)]
    # b = [e[1] for e in a if e is not None]
    # print(b)
    # ndoms = split_single(doms[0], d)
    # print(ndoms[0].pos_id)
    # ndoms[0].assign_batch_id(1)
    # print(ndoms[0].pos_id)
    # print(doms[1].pos_id)
    # r = make_domain_batch(ndoms)
    r = make_domain_batch([doms[0], doms2[0]])
    print(r)



