import torch
import pickle
import os

def specific_layer_nodes(layer, cur, depths, degrees, K):
    """ get top-K sub top layer nodes """
    # print(depths.shape, degrees.shape)
    
    # Ensure mask has the same dtype and device as degrees
    mask = torch.zeros_like(degrees, dtype=torch.float)
    
    # Get indices where depth matches the specified layer, up to cur
    indices = [i for i in range(cur) if depths[i] == layer]
    
    # Adjust K if there are fewer matching indices than K
    if len(indices) < K:
        K = len(indices)
    
    # If no indices found, return mask with only root node
    if K == 0:
        mask[0] = 1
        return mask
    
    # Convert indices to tensor and get corresponding degrees
    indices_tensor = torch.tensor(indices, device=degrees.device)
    selected_elements = degrees[indices_tensor]

    # Use torch.topk to get the indices of the top K values
    _, top_k_indices = torch.topk(selected_elements, K)

    # Get the original indices of the top K values
    original_indices = indices_tensor[top_k_indices]
    
    # Set mask values
    mask[original_indices] = 1
    mask[0] = 1  # add root node
    
    return mask


def build_parent_index(source: torch.Tensor) -> dict:
    """
    构造 parent -> index 的映射（提升查找效率）
    """
    return {int(node): idx for idx, node in enumerate(source.tolist())}

def traverse_path_nodes(source: torch.Tensor, target: torch.Tensor, cur: int, parent_dict: dict) -> torch.Tensor:
    """
    获取从根节点到当前节点路径上的节点集合
    """
    indices = []
    cur_node = cur
    while cur_node != 0:
        if cur_node not in parent_dict:
            break
        indices.append(cur_node)
        cur_node = source[parent_dict[cur_node]]
    indices.append(0)

    mask = torch.zeros(len(source) + 1, dtype=torch.float, device=source.device)
    mask[indices] = 1
    return mask

def construct_mask(edge_index: torch.Tensor, depths: torch.Tensor, focus_nodes: int = 5) -> tuple:
    """
    构造 subtop 和 trail 的掩码
    """
    device = depths.device
    n = len(depths)

    edge_index = edge_index.to(device).t().contiguous()
    source, target = edge_index[0], edge_index[1]
    degree_matrix = torch.zeros((n, n), dtype=torch.float, device=device)
    subtop_mask = torch.zeros((n, n), dtype=torch.float, device=device)
    trail_mask = torch.zeros((n, n), dtype=torch.float, device=device)

    parent_dict = build_parent_index(source)

    for j in range(len(source)):
        s, t = source[j], target[j]
        cur_depth = depths[j]

        if j > 0:
            degree_matrix[j] = degree_matrix[j - 1]
            degree_matrix[j, s] += 1

            subtop_mask[j] = specific_layer_nodes(1, cur_depth, depths, degree_matrix[j], focus_nodes)
            trail_mask[j] = traverse_path_nodes(source, target, int(t.item()), parent_dict)

    return subtop_mask, trail_mask

def save_masks(data_path: str, save_path: str, K: int = 5):
    """
    主函数：处理多个图，保存掩码
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    edges = data['edge_index']
    depths = data['depth']

    subtops, trails = [], []
    for i in range(len(edges)):
        print(f"Processing graph {i + 1}/{len(edges)}")

        edge_index = torch.tensor(edges[i], dtype=torch.long, device=device)
        depth = torch.tensor(depths[i], dtype=torch.long, device=device)

        subtop_mask, trail_mask = construct_mask(edge_index, depth, K)
        subtops.append(subtop_mask.cpu())
        trails.append(trail_mask.cpu())

    masks = {
        'subtop_mask': subtops,
        'trail_mask': trails,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(masks, f)

    print(f"Masks saved to {save_path}")

if __name__ == "__main__":
    dataname = 'pheme'
    mode = 'train'

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cur_dir, 'data', mode, f"{dataname}_features.pkl")
    save_path = os.path.join(cur_dir, 'data', mode, f"{dataname}_masks.pkl")

    save_masks(data_path, save_path)
