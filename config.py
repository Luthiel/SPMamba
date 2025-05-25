import os
import argparse
import torch


class Config:
    def __init__(self, **kwargs):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataname = kwargs.get('dataname', 'pheme')
        self.root_path = kwargs.get('root_path', os.path.join(cur_dir, 'data'))
        
        self.batch_size = kwargs.get('batch_size', 32)
        self.num_workers = kwargs.get('num_workers', 4)
        self.lr = kwargs.get('lr', 1e-5)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.epochs = kwargs.get('epochs', 50)
        self.save_path = kwargs.get('save_path', os.path.join(cur_dir, 'ckpt', self.dataname + '-SPMamba-Ext.pt'))
        self.log_path = kwargs.get('log_path', os.path.join(cur_dir, 'log'))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.parser = argparse.ArgumentParser(description='SPMamba')
        
    def parse_args(self):
        self.parser.add_argument('--root_path', type=str, default=self.root_path, help='root path')
        self.parser.add_argument('--save_path', type=str, default=self.save_path, help='save path')
        self.parser.add_argument('--log_path', type=str, default=self.log_path, help='log path')
        self.parser.add_argument('--dataname', type=str, default=self.dataname, help='dataset name')

        self.parser.add_argument('--d_input', type=int, default=768, help='input dimension')
        self.parser.add_argument('--d_hidden', type=int, default=384, help='hidden dimension')
        self.parser.add_argument('--d_state', type=int, default=16, help='state dimension')
        self.parser.add_argument('--d_model', type=int, default=768, help='model dimension')
        self.parser.add_argument('--q_head', type=int, default=16, help='query head')
        self.parser.add_argument('--kv_head', type=int, default=4, help='key-value head')
        self.parser.add_argument('--K', type=int, default=5, help='cross attention node number')
        self.parser.add_argument('--n_layer', type=int, default=2, help='number of layers')
        self.parser.add_argument('--n_class', type=int, default=4, help='class number')

        self.parser.add_argument('--num_workers', type=int, default=self.num_workers, help='num workers')
        self.parser.add_argument('--epochs', type=int, default=self.epochs, help='epochs')
        self.parser.add_argument('--batch_size', type=int, default=self.batch_size, help='batch size')
        self.parser.add_argument('--lr', type=float, default=self.lr, help='learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=self.weight_decay, help='weight decay')
        self.parser.add_argument('--device', type=str, default=self.device, help='device')
        self.parser.add_argument('--use_time_enc', type=bool, default=True, help='if use time encoding')

        args = self.parser.parse_args()
        return args