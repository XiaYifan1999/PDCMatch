import numpy as np
import torch


# def get_neigh(faces):
#     adjacent = {}

#     for triangle in faces[0]:
#         for i in range(3):
#             point = triangle[i].item()
#             if point not in adjacent:
#                 adjacent[point] = set()
#             adjacent[point].update(triangle[:i].tolist() + triangle[i + 1 :].tolist())

#     result = {point: list(neighbors) for point, neighbors in adjacent.items()}

#     return result


def get_neigh(faces):
    device = faces.device
    triangles = faces.squeeze(0)  # [9998, 3]

    # 生成所有邻接对（含双向）
    row = triangles[:, [0, 0, 1, 1, 2, 2]].flatten()  # A-A-B-B-C-C
    col = triangles[:, [1, 2, 0, 2, 0, 1]].flatten()  # B-C-A-C-A-B

    # 去重处理（参考网页4的索引表优化）
    pairs = torch.stack([row, col], dim=0).unique(dim=1)

    # 构建COO稀疏矩阵（参考网页6的稀疏矩阵构建）
    max_idx = triangles.max().item() + 1
    indices = pairs
    values = torch.ones(indices.size(1), device=device)
    sparse_mat = torch.sparse_coo_tensor(
        indices=indices, values=values, size=(max_idx, max_idx), device=device
    ).coalesce()

    # 转换为邻接字典（优化CPU-GPU交互）
    indices = sparse_mat.indices().cpu()
    row_indices = indices[0]
    col_indices = indices[1]

    # 向量化构建字典（参考网页5的并行索引策略）
    unique_rows, inverse, counts = row_indices.unique(
        return_inverse=True, return_counts=True
    )
    split_cols = torch.split(col_indices, counts.tolist())

    return {k.item(): v.tolist() for k, v in zip(unique_rows, split_cols)}


""" def comput_localscore(x_neigh, y_neigh, Pxy):

    T = torch.argmax(Pxy.squeeze(0), dim=1)
    N = len(T)
    device = Pxy.device

    com_scores = torch.zeros(N, device=device)

    for i in range(N):
        neigh_x = x_neigh.get(i, [])
        mapped_neigh_x = [T[j].item() for j in neigh_x]
        neigh_y = y_neigh.get(T[i].item(), [])

        set_mapped_x = set(mapped_neigh_x)
        set_y = set(neigh_y)
        intersection = len(set_mapped_x & set_y)

        total = len(mapped_neigh_x)
        if total == 0:
            com_scores[i] = 0.0
        else:
            com_scores[i] = intersection / total

    fscores = torch.zeros(N, device=device)
    for i in range(N):
        neigh_x = x_neigh.get(i, [])
        if not neigh_x:
            fscores[i] = 0.0
            continue

        neighbor_scores = com_scores[neigh_x]
        mean_neighbor_score = (
            neighbor_scores.mean() if len(neighbor_scores) > 0 else 0.0
        )

        fscores[i] = 0.5 * (com_scores[i] + mean_neighbor_score)

    return fscores """


def comput_localscore(x_neigh, y_neigh, Pxy):
    device = Pxy.device
    T = torch.argmax(Pxy.squeeze(0), dim=1)
    N = T.shape[0]
    
    def build_coo_adj(neigh_dict, max_idx, device):
        rows, cols = [], []
        for k, v in neigh_dict.items():
            rows.extend([k] * len(v))
            cols.extend(v)

        # 将列表转换为Tensor
        # cols_tensor = torch.tensor(cols, dtype=torch.long, device=device)
        indices = torch.tensor([rows, cols], dtype=torch.long, device=device)

        # 使用cols_tensor生成ones_like
        values = torch.ones(len(cols), dtype=torch.float32, device=device)  # 正确用法

        return torch.sparse_coo_tensor(indices, values, (max_idx, max_idx)).coalesce()

    # 构建邻接矩阵
    x_adj = build_coo_adj(x_neigh, max(x_neigh.keys()) + 1 if x_neigh else 0, device)
    y_adj = build_coo_adj(y_neigh, max(y_neigh.keys()) + 1 if y_neigh else 0, device)

    row_indices = x_adj.indices()[0]
    row_counts = row_indices.bincount(minlength=N)
    ptr = torch.cat([torch.tensor([0], device=device), row_counts.cumsum(dim=0)])

    # 批量计算交集（参考网页3的向量化方法）
    mapped_nodes = T[x_adj.indices()[1]]  # 映射后的邻居
    y_mask = y_adj.to_dense()[T]  # 目标节点的邻居掩码
    valid_mask = y_mask[:, mapped_nodes]  # 有效性布尔矩阵

    intersection = torch.zeros(N, device=device)
    for i in range(N):
        start = ptr[i].item()  # 必须使用.item()
        end = ptr[i + 1].item()  # 必须使用.item()
        intersection[i] = valid_mask[i, start:end].sum()

    # 计算com_scores（参考网页5的高效计算）
    total = row_counts.float().clamp_min(1e-8)
    com_scores = intersection / total

    # 计算fscores（参考网页2的稀疏矩阵运算）
    neighbor_sum = torch.sparse.mm(x_adj, com_scores.unsqueeze(1)).squeeze()
    fscores = 0.5 * (com_scores + neighbor_sum / total)

    return fscores

