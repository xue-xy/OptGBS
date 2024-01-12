import torch


def evaluate_inf_min_arg(region_min, region_max, c: torch.Tensor):
    if c.dim() == 1:
        c = torch.unsqueeze(c, dim=0)
    x = torch.where(c > 0, region_min, region_max)

    return torch.sum(x * c, dim=[e for e in range(1, c.dim())]), x


def evaluate_inf_max_arg(region_min, region_max, c: torch.Tensor):
    if c.dim() == 1:
        c = torch.unsqueeze(c, dim=0)
    x = torch.where(c > 0, region_max, region_min)

    return torch.sum(x * c, dim=[e for e in range(1, c.dim())]), x


if __name__ == '__main__':
    hsize = (2, 3, 3)
    rmin = torch.zeros(hsize)
    rmax = torch.ones(hsize)

    coeff = torch.zeros(hsize)
    coeff[0] = -0.5
    coeff[1] = 0.5
    coeff = torch.reshape(coeff, (1, 2, 3, 3))
    results, x = evaluate_inf_max_arg(rmin, rmax, coeff)
    print(x)
