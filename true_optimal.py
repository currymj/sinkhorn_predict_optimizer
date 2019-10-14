import gurobipy as gp
import torch

def opt_match(cost_matrix):
    # this will have to be modified to handle multiple similar items (integer not binary)
    model = gp.Model()
    x = {}
    l_n = cost_matrix.shape[0]
    r_n = cost_matrix.shape[1]
    for i in range(l_n):
        for j in range(r_n):
            x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f'x_{i}_{j}')
    model.update()
    match_once_constraints = []
    for i in range(l_n):
        match_once_constraints.append(
            model.addConstr(gp.quicksum(x[i, j] for j in range(r_n)) <= 1))
    for j in range(r_n):
        match_once_constraints.append(
            model.addConstr(gp.quicksum(x[i, j] for i in range(l_n)) <= 1))
    obj = gp.QuadExpr()
    for i in range(l_n):
        for j in range(r_n):
            obj += x[i, j] * cost_matrix[i, j].item()
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    model.optimize()
    model.update()

    result = torch.zeros_like(cost_matrix)
    for i in range(l_n):
        for j in range(r_n):
            result[i, j] = x[i, j].x
    return result
