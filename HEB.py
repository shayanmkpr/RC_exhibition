import torch

def corr_mean(x):
    mean_row = x@(torch.ones(x[0].size(0)).to(device=x.device))
    mean_tensor = mean_row.repeat(x[0].size(0) , 1).t()
    return mean_tensor.to(device = x.device)

def corr_std(corr):
    std = corr.diag().repeat(corr[0].size(0) , 1)
    return (std.t()).to(device = corr.device)

def pearson_correlation(x , y):     
    
    mean_x = corr_mean(x)
    mean_y = corr_mean(y)

    centered_x = x - mean_x
    centered_y = y - mean_y

    numerator = torch.matmul(centered_x , centered_y.t())

    std = corr_std(numerator) + 1e-20*torch.ones((numerator[0].size(0) , numerator[0].size(0))).to(device=x.device)

    return (numerator/torch.sqrt(std@std.t()))


def Heb_func(sigma , lr_h , p , ps , x , y ,n_nodes):

    corr = pearson_correlation(x , y)

    upper_p = torch.topk(corr.view(-1), p).values[-1]
    lower_p = torch.topk(-corr.view(-1), p).values[-1]
    active_corr = (corr >= upper_p) | (corr <= lower_p)
    active_corr = active_corr * 1 - (torch.eye(n_nodes , n_nodes).to(device = x.device))
    
    a = torch.bernoulli(torch.tensor(ps)).to(device = x.device)
    hebb = a * sigma * lr_h * corr

    delta_W = hebb * active_corr 

    return (delta_W)