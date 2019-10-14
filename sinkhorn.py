import torch

def sinkhorn_plan(weight_mat, eps=1e-1, rounds=5):
    K = torch.exp(-weight_mat / eps)
    l_n = weight_mat.shape[0]
    r_n = weight_mat.shape[1]
    a = torch.ones(l_n) / l_n
    b = torch.ones(r_n) / r_n
    v = torch.ones(l_n)
    u = a / (K @ v)
    v = b / (K.t() @ u)

    for i in range(rounds):
        u = a / (K @ v)
        v = b / (K.t() @ u)

    return torch.diag(u) @ K @ torch.diag(v)
