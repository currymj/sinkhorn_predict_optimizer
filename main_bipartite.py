from bipartite_data import generate_data_linear, data_batch_format, generate_data_rand_nn, generate_data_hide_features
from true_optimal import opt_match
from sinkhorn import sinkhorn_plan
import torch
import numpy as np
from tqdm import tqdm as tqdm
from torch import nn, optim

def trainbatch(model, criterion, optimizer, vecs, edges, device='cuda'):
    model.train()
    vecs.to(device)
    edges.to(device)
    model.zero_grad()
    out = model(vecs)
    loss = criterion(out, edges)
    save_loss = loss.item()
    loss.backward()
    optimizer.step()
    return save_loss


def trainbatch_sinkhorn(model, optimizer, vecs, true_cost, device='cuda', eps=1e-1, rounds=5):
    model.train()
    vecs.to(device)
    model.zero_grad()
    predicted_cost_mat = torch.sigmoid(model(vecs)).view(true_cost.shape[0], true_cost.shape[1])
    predicted_matching = sinkhorn_plan(-predicted_cost_mat, eps=eps, rounds=rounds)
    loss = -torch.sum(true_cost * predicted_matching)
    save_loss = -loss.item()
    loss.backward()
    optimizer.step()
    return save_loss


def prediction_train(model, vecs, edges, epochs=1000):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    losses = []
    for i in tqdm(range(epochs)):
        loss = trainbatch(model, criterion, optimizer, vecs, edges)
        losses.append(loss)
    return model, losses

def predict_optimize_train(model, vecs, edge_mat, epochs=1000, eps=1e-1, rounds=5):
    optimizer = optim.Adam(model.parameters())
    losses = []
    for i in tqdm(range(epochs)):
        loss = trainbatch_sinkhorn(model, optimizer, vecs, edge_mat, eps=eps, rounds=rounds)
        losses.append(loss)
    return model, losses


def predicted_edges_to_matrix(edges):
    dim = int(np.sqrt(edges.shape[0]))
    return edges.view(dim, dim)

def eval_true_performance(model, feats_batch, edge_mat):
    true_opt = torch.sum(opt_match(edge_mat) * edge_mat)
    model_predictions = torch.sigmoid(model(feats_batch)).view(edge_mat.shape[0], edge_mat.shape[1]).detach()
    predict_opt = torch.sum(opt_match(model_predictions) * edge_mat)
    return predict_opt, true_opt


if __name__ == '__main__':
    dim = 10
    #u_feats, v_feats, edge_mat = generate_data_linear(dim=dim, N=100)
    #u_feats, v_feats, edge_mat = generate_data_rand_nn(dim=dim, N=100)
    u_feats, v_feats, edge_mat = generate_data_hide_features(dim=dim*5, keep=dim, N=100)
    batch_x, batch_y = data_batch_format(u_feats, v_feats, edge_mat)

    predictive_model = nn.Sequential(*[nn.Linear(dim * 2, 128), nn.ReLU(), nn.Linear(128, 1)])

    true_perf_before, opt_perf = eval_true_performance(predictive_model, batch_x, edge_mat)
    print('true perf before', true_perf_before)
    print('true opt perf', opt_perf)

    trained_model, training_loss = prediction_train(predictive_model, batch_x, batch_y)

    true_perf_after, opt_perf = eval_true_performance(predictive_model, batch_x, edge_mat)
    print('true perf after', true_perf_after)

    #u_new, v_new, edge_new = generate_data_linear(dim=dim, N=100)
    #u_new, v_new, edge_new = generate_data_rand_nn(dim=dim, N=100)
    u_new, v_new, edge_new = generate_data_hide_features(dim=dim*5, keep=dim, N=100)
    batch_new, _ = data_batch_format(u_new, v_new, edge_new)
    true_perf_test, opt_perf_test = eval_true_performance(predictive_model, batch_new, edge_new)

    print('perf on unseen test', true_perf_test)
    print('unseen opt perf', opt_perf_test)

    po_model = nn.Sequential(*[nn.Linear(dim * 2, 128), nn.ReLU(), nn.Linear(128, 1)])
    po_perf_before, opt_perf = eval_true_performance(po_model, batch_x, edge_mat)
    print('predict/optimize perf before training', po_perf_before)

    trained_po_model, training_loss = predict_optimize_train(po_model, batch_x, edge_mat, rounds=10)
    po_perf_after, _ = eval_true_performance(po_model, batch_x, edge_mat)
    print('predict/optimize perf after training', po_perf_after)

    true_po_test, opt_perf_test = eval_true_performance(po_model, batch_new, edge_new)
    print('predict/optimize unseen test perf', true_po_test)

