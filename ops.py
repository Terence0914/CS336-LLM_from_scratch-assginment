import torch

def softmax(in_features, dim):
    # m = max_j(v_j)：沿 dim 找最大值；保留该维度以便后续广播。
    largest = in_features.max(dim=dim, keepdim=True).values
    # z_i = v_i - m：平移分数，使最大值变成 0，避免计算 exp(大数) 时溢出。
    stabilized = in_features - largest
    # n_i = exp(z_i) = exp(v_i - m)：将平移后的每个分数转换为正数。
    exponentials = torch.exp(stabilized)
    # Z = sum_j exp(v_j - m)：沿同一个 dim 计算 Softmax 的归一化分母。
    normalizer = exponentials.sum(dim=dim, keepdim=True)
    # softmax(v)_i = exp(v_i - m) / sum_j exp(v_j - m)
    return exponentials / normalizer