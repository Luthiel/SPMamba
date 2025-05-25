from collections import deque

import torch
from typing import List
import numpy as np

class TreeNode(object):
    def __init__(self, idx=None):
        self.idx = idx
        self.children = []
        self.parent = None
        self.timestamp = None
        self.depth = None
        self.degree = 0

def construct_tree(edge_index: torch.tensor, timestamps: List[int], depths: List[int], max_size: int) -> List[TreeNode]:
    root = TreeNode(0)
    root.timestamp = timestamps[0]
    root.depth = depths[0]
    
    nodes = [root]
    cnt = 1
    print(f'edge_index {edge_index.shape}')
    for source, target in edge_index:
        cur = TreeNode(target)
        parent = nodes[source]
        cur.parent = parent
        cur.timestamp = timestamps[target]
        cur.depth = depths[target]
        nodes.append(cur)
        parent.children.append(cur)
        parent.degree += 1
        
        cnt += 1
        if cnt == max_size:
            break
    return nodes    

def dfs(node, sequence):
    if node is None:
        return
    sequence.append(node)
    for child in node.children:
        dfs(child, sequence)

def bfs(node):
    if not node:
        return
    sequence = []
    queue = deque([node])
    while queue:
        node = queue.popleft()
        sequence.append(node)
        queue.extend(node.children)
    return sequence

def pad_or_trunc_idx(sequence, max_size):
    if len(sequence) > max_size:
        return sequence[:max_size]
    else:
        offset = max_size - len(sequence)
        prefix = np.arange(offset)
        offset_sequence = np.array(sequence) + offset
        res = np.concatenate([prefix, offset_sequence])
        return  res

def construct_tree_simple(edge_index: torch.tensor, max_size: int) -> List[TreeNode]:
    root = TreeNode(0)
    edge_index = edge_index.t()
    
    nodes = [root]
    cnt = 1
    print(f'edge_index {edge_index.shape}')
    for source, target in zip(edge_index[0], edge_index[1]):
        cur = TreeNode(target)
        parent = nodes[source]
        cur.parent = parent
        nodes.append(cur)
        parent.children.append(cur)
        
        cnt += 1
        if cnt == max_size:
            break
    return nodes  

def sequentialize(edge_index, max_size, return_idx=True):
    sequence = construct_tree_simple(edge_index, max_size)
    root = sequence[0]
    dfs_sequence = []
    dfs(root, dfs_sequence)
    bfs_sequence = bfs(root)
    
    # deg_sequence = sorted(sequence, key=lambda x: len(x.children), reverse=True)

    if return_idx:
        sequence = [node.idx for node in sequence]
        dfs_sequence = [node.idx for node in dfs_sequence]
        bfs_sequence = [node.idx for node in bfs_sequence]
        
        # print('sequence', sequence)
        # print('dfs_sequence', dfs_sequence)
        # print('bfs_sequence', bfs_sequence)
        
        sequence = pad_or_trunc_idx(sequence, max_size)
        dfs_sequence = pad_or_trunc_idx(dfs_sequence, max_size)
        bfs_sequence = pad_or_trunc_idx(bfs_sequence, max_size)
        
        return dfs_sequence, bfs_sequence
    
    
    return sequence, dfs_sequence, bfs_sequence
