import numpy as np
import torch
from tqdm import tqdm as tqdm
from true_optimal import opt_match
import pickle
import matplotlib.pyplot as plt
from torch import optim, nn
from sinkhorn import sinkhorn_plan

def trainbatch_sinkhorn(model, optimizer, vecs, true_cost, device='cuda', eps=1e-1, rounds=5):
    model.train()
    #vecs.to(device)
    model.zero_grad()
    predicted_cost_mat = torch.sigmoid(model(vecs)).view(true_cost.shape[0], true_cost.shape[1])
    predicted_matching = sinkhorn_plan(-predicted_cost_mat, eps=eps, rounds=rounds)
    loss = -torch.sum(true_cost * predicted_matching)
    save_loss = -loss.item()
    loss.backward()
    optimizer.step()
    return save_loss
def trainbatch(model, criterion, optimizer, vecs, edges, device='cuda'):
    model.train()
    #vecs.to(device)
    #edges.to(device)
    model.zero_grad()
    out = model(vecs)
    loss = criterion(out, edges)
    save_loss = loss.item()
    loss.backward()
    optimizer.step()
    return save_loss

def prediction_train(model, optimizer, vecs, edges, criterion):
    losses = []
    loss = trainbatch(model, criterion, optimizer, vecs, edges)
    return model, loss

def predict_optimize_train(model, optimizer, vecs, edge_mat, eps=1e-1, rounds=5):
    loss = trainbatch_sinkhorn(model, optimizer, vecs, edge_mat, eps=eps, rounds=rounds)
    return model, loss

def eval_true_performance(model, feats_batch, edge_mat):
    true_opt = torch.sum(opt_match(edge_mat) * edge_mat)
    model_predictions = torch.sigmoid(model(feats_batch)).view(edge_mat.shape[0], edge_mat.shape[1]).detach()
    predict_opt = torch.sum(opt_match(model_predictions) * edge_mat)
    return predict_opt, true_opt

def eval_loss(model, feats_batch, edges_batch):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    out = model(feats_batch)
    loss = criterion(out, edges_batch)
    return loss

if __name__ == '__main__':
    with open('cora_data.pickle', 'rb') as datafile:
        edge_instances, feature_instances = pickle.load(datafile)

    dim = feature_instances.shape[2]
    num_elems = int(np.sqrt(feature_instances.shape[1]))
    predictive_model = nn.Sequential(*[nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128, 1)])
    pred_criterion = nn.BCEWithLogitsLoss()
    pred_optimizer = optim.Adam(predictive_model.parameters(), lr=1e-4, weight_decay=1e-3)

    po_model = nn.Sequential(*[nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128, 1)])
    po_optimizer = optim.Adam(po_model.parameters(), lr=1e-4, weight_decay=1e-3)

    for epochs in tqdm(range(100)):
        for ind in range(5):
            batch_x = torch.from_numpy(feature_instances[ind]).float()
            batch_y = torch.from_numpy(edge_instances[ind]).float().view(-1,1)
            edge_mat = batch_y.view(num_elems,num_elems)

            prediction_train(predictive_model, pred_optimizer, batch_x, batch_y, pred_criterion)

            predict_optimize_train(po_model, po_optimizer, batch_x, edge_mat, rounds=10)


    for ind in range(5,10):
        batch_x = torch.from_numpy(feature_instances[ind]).float()
        batch_y = torch.from_numpy(edge_instances[ind]).float().view(-1,1)
        edge_mat = batch_y.view(num_elems,num_elems)

        true_po_test, opt_perf_test = eval_true_performance(po_model, batch_x, edge_mat)
        print('predict/optimize unseen test perf', true_po_test)
        print('opt unseen test perf', opt_perf_test)

        true_perf_test, _ = eval_true_performance(predictive_model, batch_x, edge_mat)

        print('perf on unseen test', true_perf_test)

        print('two-stage prediction error', eval_loss(predictive_model, batch_x, batch_y))
        print('p&o prediction error', eval_loss(po_model, batch_x, batch_y))
