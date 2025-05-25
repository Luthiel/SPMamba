import os
import torch
import torch.nn.functional as F
import numpy as np
import json

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from time import time
from encoder import MPNet
from math import exp
import pickle
from sequence import sequentialize


class GraphBuilder:
    def __init__(self, contents, events, labels, dataname, use_time_enc=True):
        
        self.encoder = MPNet(use_time_enc)
        self.contents = contents
        self.events = events
        self.labels = labels
        self.dataname = dataname
        
    def _get_edge_weight(self, node_feats, edges, times, depths, alpha=.5, beta=1., scale=1., use_time_bias=True):
        node_num = len(node_feats)
        
        adj_mask = torch.zeros((node_num, node_num))
        decay_factor = torch.zeros((node_num, node_num))
        max_time, min_time = max(times), min(times)
        
        for row, col in zip(edges[0], edges[1]):
            adj_mask[row][col] = 1
            delta_time = float((times[col] - min_time) / (max_time - min_time) * scale)
            if use_time_bias:
                decay_factor[row][col] = exp(-alpha * depths[row]) + self._get_time_bias(delta_time)
            else:
                decay_factor[row][col] = exp(-alpha * depths[row] - beta * delta_time)
            
        sim_matrix = torch.matmul(node_feats, node_feats.t())
        edge_weights = sim_matrix * adj_mask * decay_factor
        
        weight_lst = []
        for row, col in zip(edges[0], edges[1]):
            weight_lst.append(edge_weights[row][col])
        
        return weight_lst
    
    def _get_time_bias(self, time_diffs):
        epsilon = 1
        gamma = 0.3
        delta = 0.5
        return -epsilon * max(0, time_diffs) + gamma * exp(-(time_diffs * time_diffs) / (2 * delta ** 2))

    def _check_smooth(self, nodes: list):
        if max(nodes) > len(nodes) - 1:
            return False
        return True
        
    def build_graph(self, event_id) -> Data:
        # print(f'contenss type is {type(self.contents)}')
        data = self.contents[event_id]
        conversation, edge_index, edge_weights, temporals, depths = [], [[], []], [], [], []
        for post_id in data.keys():
            attr = data[post_id]
            parent_id = attr['parent_id']
            if parent_id is not None:
                edge_index[0].append(int(post_id))
                edge_index[1].append(int(attr['parent_id']))
            
            conversation.append(attr['content'])
            temporals.append(attr['timestamp'])
            depths.append(attr['depth'])
        
        if not self._check_smooth(edge_index[0]):
            edge_index[0] = [i for i in range(len(edge_index[0]))]

        temporals = torch.tensor(temporals, dtype=torch.float).view(-1, 1)
        node_features = self.encoder(conversation, temporals)
        # print(f"node features shape: {node_features.shape}")
        # print(f"edge index shape: {len(edge_index[0])}, {len(edge_index[1])}")
        edge_weights = self._get_edge_weight(node_features, edge_index, temporals, depths)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() # shape: [2, N]
        edge_weights = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
        depths = torch.tensor(depths, dtype=torch.long).view(-1, 1)
        x = torch.tensor(node_features, dtype=torch.float).clone().detach()
        label = self.labels[event_id]

        return x, edge_index, edge_weights, depths, label

    def build_batch_graphs(self, event_ids, max_size=100):
        x_batch, index_batch, weights_batch, label_batch, depth_batch, dfs_batch, bfs_batch = [], [], [], [], [], [], []
        cnt = 0
        for eid in event_ids:
            if len(self.contents[eid].keys()) < 10:
                continue
            cnt += 1
            x, edge_index, edge_weights, depths, label = self.build_graph(eid)
            dfs, bfs = sequentialize(edge_index, max_size)
            x, edge_index, edge_weights, depths = self.sample_or_pad(x, edge_index, edge_weights, depths, max_size)
            
            x_batch.append(x)
            index_batch.append(edge_index)
            weights_batch.append(edge_weights)
            depth_batch.append(depths)
            label_batch.append(label)
            dfs_batch.append(dfs)
            bfs_batch.append(bfs)
        
        print(f"final valid events: {cnt}")
        
        return x_batch, index_batch, weights_batch, depth_batch, label_batch, dfs_batch, bfs_batch

    def sample_or_pad(self, x, edge_index, edge_weights, depths, max_size, pad_value=0):
        seq_len, _ = x.shape
        
        if seq_len == max_size:
            return x, edge_index, edge_weights, depths
        elif seq_len < max_size:
            padding_size = max_size - seq_len
            # pad position -> left, right, top, bottom
            padded_x = F.pad(x, (0, 0, padding_size, 0), value=pad_value)
            padded_edge_index = F.pad(edge_index, (0, 0, padding_size, 0), value=pad_value)
            padded_edge_weights = F.pad(edge_weights, (0, 0, padding_size, 0), value=pad_value)
            padded_depths = F.pad(depths, (0, 0, padding_size, 0), value=pad_value)
            return padded_x, padded_edge_index, padded_edge_weights, padded_depths
        else:
            truncated_x = x[:max_size, :]
            truncated_edge_index = edge_index[:max_size - 1, :]
            truncated_edge_weights = edge_weights[:max_size - 1, :]
            truncated_depths = depths[:max_size, :]
            return truncated_x, truncated_edge_index, truncated_edge_weights, truncated_depths

class GraphDataset(Dataset):
    def __init__(self, data_path, mask_path=None, use_multi_seq=False, dataname='weibo'):
        assert os.path.exists(data_path), f"Data path {data_path} does not exist."
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        
        self.use_multi_seq = use_multi_seq
        self.data = data['x']
        self.indices = data['edge_index']
        self.weights = data['edge_weights']
        self.depths = data['depth']
        self.labels = data['label']
        if dataname == 'pheme': # 冲转为二分类
            class_2 = []
            for label in self.labels:
                if label == [1, 0, 0, 0] or label == [0, 0, 1, 0]:
                    class_2.append([1, 0])
                else:
                    class_2.append([0, 1])
            self.labels = class_2
        # self.bfs_seq = data['bfs']
        # self.dfs_seq = data['dfs']
        
        self.mask_path = mask_path
        self.subtop_mask = None
        self.trail_mask = None
        if mask_path:
            with open(mask_path, 'rb') as file:
                masks = pickle.load(file)
                self.subtop_mask = masks['subtop_mask']
                self.trail_mask = masks['trail_mask']
        
    def __getitem__(self, idx):
        if self.mask_path:
            return self.data[idx], self.indices[idx], self.weights[idx], self.depths[idx], \
                self.labels[idx], self.subtop_mask[idx], self.trail_mask[idx]
                
        return self.data[idx], self.indices[idx], self.weights[idx], self.depths[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

def read_data(path):
    events, labels = [], {}
    onehot_map = {'0': [1, 0, 0, 0], '1': [0, 1, 0, 0], '2': [0, 0, 1, 0], '3': [0, 0, 0, 1]}
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            item = line.strip().split('\t')
            eid, label = item[0], item[1]
            events.append(eid)
            labels[eid] = onehot_map[label]
        file.close()
        
    return events, labels

def load_data(root_path, dataname, batch_size=64, use_time_enc=True):
    if use_time_enc:
        train_path = os.path.join(root_path, 'train', dataname + '_features.pkl')
        dev_path = os.path.join(root_path, 'dev', dataname + '_features.pkl')
        test_path = os.path.join(root_path, 'test', dataname + '_features.pkl')
    else:
        train_path = os.path.join(root_path, 'train', dataname + '_features_without_time.pkl')
        dev_path = os.path.join(root_path, 'dev', dataname + '_features_without_time.pkl')
        test_path = os.path.join(root_path, 'test', dataname + '_features_without_time.pkl')
    
    train_mask_path = os.path.join(root_path, 'train', dataname + '_masks.pkl')
    dev_mask_path = os.path.join(root_path, 'dev', dataname + '_masks.pkl')
    test_mask_path = os.path.join(root_path, 'test', dataname + '_masks.pkl')
    
    # print(f"Loading {dataname} dataset..., data path constructed")

    train_data = GraphDataset(train_path, mask_path=train_mask_path, dataname=dataname)
    dev_data = GraphDataset(dev_path, mask_path=dev_mask_path, dataname=dataname)
    test_data = GraphDataset(test_path, mask_path=test_mask_path, dataname=dataname)
    
    print(f"Loading {dataname} dataset..., data loaded")
    
    train_iter = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True)
    dev_iter = DataLoader(dev_data, batch_size=batch_size, num_workers=4, shuffle=False)
    test_iter = DataLoader(test_data, batch_size=batch_size, num_workers=4, shuffle=False)

    return train_iter, dev_iter, test_iter

def save(contents, events, labels, dataname, save_path, use_time_enc):
    print(f'total events: {len(events)}')
    max_size = 300 if dataname == 'weibo' else 50
    start = time()
    x_batch, index_batch, weights_batch, depth_batch, label_batch, dfs_batch, bfs_batch = \
            GraphBuilder(contents, events, labels, dataname, use_time_enc).build_batch_graphs(events, max_size)
          
    obj = {'x': x_batch, 
           'edge_index': index_batch, 
           'edge_weights': weights_batch, 
           'depth': depth_batch, 
           'label': label_batch, 
           'dfs': dfs_batch,
           'bfs': bfs_batch,
           }  
    
    with open(save_path, 'wb') as file:
        pickle.dump(obj, file)
        
    end = time()
    print(f"save {dataname} data done! Cost time: {end - start:.2f}s")

if __name__ == "__main__":
    dataname = 'pheme'
    cur_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    train_path = os.path.join(cur_dir, 'train', dataname + '.txt')
    dev_path = os.path.join(cur_dir, 'dev', dataname + '.txt')
    test_path = os.path.join(cur_dir, 'test', dataname + '.txt')

    train_events, train_labels = read_data(train_path)
    dev_events, dev_labels = read_data(dev_path)
    test_events, test_labels = read_data(test_path)
    
    with open(os.path.join(cur_dir, dataname + '_new_content_2.json'), 'r', encoding='utf-8') as file:
        contents = json.load(file)
        
    save(contents, train_events, train_labels, dataname, save_path=os.path.join(cur_dir, 'train', dataname + '_features_without_time.pkl'), use_time_enc=False)
    save(contents, dev_events, dev_labels, dataname, save_path=os.path.join(cur_dir, 'dev', dataname + '_features_without_time.pkl'), use_time_enc=False)
    save(contents, test_events, test_labels, dataname, save_path=os.path.join(cur_dir, 'test', dataname + '_features_without_time.pkl'), use_time_enc=False)
