import torch
from hebbian_GPU import pearson_correlation

def oja(sigma, lr_h, p, ps, x, y, n_nodes, model):
    corr = pearson_correlation(x, y)

    upper_p = torch.topk(corr.view(-1), p).values[-1]
    lower_p = torch.topk(-corr.view(-1), p).values[-1]
    active_corr = (corr >= upper_p) | (corr <= lower_p)
    active_corr = active_corr.float() * (1 - torch.eye(n_nodes, n_nodes))
    
    a = torch.bernoulli(torch.full((n_nodes, n_nodes), ps))

    delta_W = a * sigma * lr_h * (corr - (pearson_correlation(x , y) / torch.norm( y)))
    delta_W = delta_W * active_corr
    
    return delta_W