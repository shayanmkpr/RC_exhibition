import torch

def pearson_correlation(x, y):
  mean_x = torch.mean(x)
  mean_y = torch.mean(y)
  centered_x = x - mean_x
  centered_y = y - mean_y

  std_x = torch.dot(centered_x , centered_x)
  std_y = torch.dot(centered_y , centered_y)
  numerator = torch.dot(centered_x , centered_y)
  return (numerator/(std_x * std_y)).to(device = x.device)

def Heb_func(sigma , lr_h , p , ps , x , n_nodes):

  delta_W = torch.zeros(n_nodes , n_nodes)
  corr = torch.zeros(n_nodes , n_nodes)

  for i in range(n_nodes):
    for j in range(n_nodes):
      if i==j:
        a=0
        corr[i][j] = 0
      else:
        corr[i][j] = pearson_correlation(x[i] , x[j])

  upper_p = torch.topk(corr.view(-1), p).values[-1]
  lower_p = torch.topk(-corr.view(-1), p).values[-1]
  active_corr = (corr >= upper_p) | (corr <= lower_p)

  for i in range(n_nodes):
    for j in range(n_nodes):
  
      if i==j:
        a=0
        delta_W[i][j] = 0
  
      else:
        a = torch.bernoulli(torch.tensor(ps))
        if active_corr[i][j]:
          delta_W[i , j] = a * sigma * lr_h * corr[i][j]
  return (delta_W).to(device = x.device)

# x = torch.tensor([[-0.0210, -0.1050, -0.1919, -0.2821, -0.3754],
#         [-0.0380, -0.2089, -0.3796, -0.5526, -0.7273],
#         [ 2.5694,  2.6233,  2.6742,  2.7241,  2.7729],
#         [ 2.8581,  2.8401,  2.8239,  2.8081,  2.7928],
#         [-1.6013, -1.6483, -1.6953, -1.7403, -1.7830],
# ])
# # print(Heb_func())

# print(pearson_correlation(x))