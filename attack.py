import torch
from convback import VMODEL


def pgd_attack(vmodel: VMODEL, x: torch.Tensor, c, eps, norm='inf'):
    sus = x.clone()
    if sus.dim() == 3:
        sus = sus.unsqueeze(dim=0)

    if norm == 'inf':
        xmin = torch.clip(x - eps, min=0.0)
        xmax = torch.clip(x + eps, max=1.0)

    found = False

    ite_time = 40
    lr = 0.05

    for i in range(ite_time):
        sus.requires_grad = True

        pred = vmodel.model(sus)
        loss = torch.sum(pred * c)
        # print(loss)
        if loss.detach() < 0:
            found = True
            break

        loss.backward()

        sus = sus - lr * sus.grad

        sus = sus.detach()
        sus = torch.maximum(sus, xmin)
        sus = torch.minimum(sus, xmax)
        # print(torch.sum(vmodel.model(sus) * c))
        # print('-'*40)

    return found


if __name__ == '__main__':
    pass