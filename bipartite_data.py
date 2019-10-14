import torch

def generate_data_linear(dim = 10, N = 100):
    u_feats = torch.rand(N, dim)
    v_feats = torch.rand(N, dim)

    scores_mat = u_feats @ (v_feats.t())

    edge_mat = (scores_mat > 2.5).float()

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