import numpy as np
import torch
from tqdm import tqdm as tqdm
import pickle
import matplotlib.pyplot as plt
from torch import optim, nn

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

def prediction_train(model, vecs, edges, epochs=1000):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    losses = []
    for i in tqdm(range(epochs)):
        loss = trainbatch(model, criterion, optimizer, vecs, edges)
        losses.append(loss)
    return model, losses

if __name__ == '__main__':
    with open('cora_data.pickle', 'rb') as datafile:
        edge_instances, feature_instances = pickle.load(datafile)

    dim = feature_instances.shape[2]

    batch_x = torch.from_numpy(feature_instances[0]).float()
    batch_y = torch.from_numpy(edge_instances[0]).float().view(-1,1)


    predictive_model = nn.Sequential(*[nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 1)])

    trained_model, training_loss = prediction_train(predictive_model, batch_x, batch_y)


