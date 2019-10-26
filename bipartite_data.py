import torch
from torch import nn

def generate_data_linear(dim = 10, N = 100):
    u_feats = torch.rand(N, dim)
    v_feats = torch.rand(N, dim)

    scores_mat = u_feats @ (v_feats.t())

    edge_mat = (scores_mat > scores_mat.median()).float()

    return u_feats, v_feats, edge_mat

def generate_data_linear_hide_features(dim=10, keep=5, N=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    u_feats = torch.rand(N, dim, device=device)
    v_feats = torch.rand(N, dim, device=device)
    scores_mat = u_feats @ (v_feats.t())

    edge_mat = (scores_mat > scores_mat.median()).float()

    u_hidden = u_feats[:,0:keep]
    v_hidden = v_feats[:,0:keep]

    return u_hidden, v_hidden, edge_mat

def generate_data_hide_features(rand_nn, dim=10, keep=5, N=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    u_feats = torch.rand(N, dim, device=device)
    v_feats = torch.rand(N, dim, device=device)
    scores_mat = rand_nn(u_feats) @ (rand_nn(v_feats).t())
    edge_mat = (scores_mat > scores_mat.median()).float()

    u_hidden = u_feats[:,0:keep]
    v_hidden = v_feats[:,0:keep]

    return u_hidden, v_hidden, edge_mat


def generate_data_rand_nn(rand_nn, dim = 10, N = 100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    u_feats = torch.rand(N, dim, device=device)
    v_feats = torch.rand(N, dim, device=device)
    scores_mat = rand_nn(u_feats) @ (rand_nn(v_feats).t())
    edge_mat = (scores_mat > scores_mat.median()).float()
    return u_feats, v_feats, edge_mat

def data_batch_format(u_feats, v_feats, edge_mat):
    def gen(u, v, edges):
        for i in range(u.shape[0]):
            for j in range(v.shape[0]):
                yield torch.cat((u[i,:], v[j,:])), edges[i, j]

    results = list(gen(u_feats, v_feats, edge_mat))
    vecs = torch.stack([x[0] for x in results])
    edges = torch.stack([x[1] for x in results]).view(-1, 1)
    return vecs, edges
