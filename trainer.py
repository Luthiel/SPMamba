import torch
import torch.nn as nn
import numpy as np
import random
import logging

from tqdm import tqdm
from early_stopping_pytorch import EarlyStopping
from gqa import DGSL
from mamba import Mamba2Block
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from metrics import evaluate_model
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def get_logger(log_name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_name), logging.StreamHandler()]
    )
    logger = logging.getLogger('info_recorder')
    return logger

def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight)
        return output

class SPMamba(nn.Module):
    def __init__(self, in_dim, hid_dim, d_model, d_state, q_head, kv_head, n_layer, n_class, mamba_out=False):
        super().__init__()
        self.dgsl = DGSL(in_dim, hid_dim, d_model, q_head, kv_head)
        print("DGSL initialized!")
        self.ln = nn.LayerNorm(in_dim) 
        if not mamba_out:
            self.m_blocks = nn.ModuleList(
                [ResidualBlock(d_model, d_state) for _ in range(n_layer)]
            )
            print("Mamba2Block initialized!")
        self.rmsn = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_class)
        self.mamba_out = mamba_out

    def forward(self, x, edge_index, mask1=None, mask2=None):
        """ x.shape = [batch_size, seq_len, features] """
        x = self.dgsl(x, edge_index, mask1, mask2)
        x = self.ln(x)
            
        if not self.mamba_out:
            for layer in self.m_blocks:
                x = layer(x)
        x = self.rmsn(x)[:, -1, :] # 取最后一个时间步的输出
        x = self.out_proj(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.mixer = Mamba2Block(d_model, d_state)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        x1 = self.norm(x)
        x2 = self.mixer(x1)
        output = x2 + x1

        return output
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor [num_classes]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)

        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)  
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)        

        if self.alpha is not None:
            alpha_t = self.alpha[targets]                        
            loss = -alpha_t * ((1 - pt) ** self.gamma) * logpt
        else:
            loss = -((1 - pt) ** self.gamma) * logpt

        return loss.mean() if self.reduction == 'mean' else loss.sum()


class Trainer:
    def __init__(self, model, epoch, lr, device, save_path, num_classes=4, use_mask=True, use_focal=True):
        self.model = model.to(device)
        self.epoch = epoch
        self.lr = lr
        self.device = device
        self.model_path = save_path
        self.use_mask = use_mask
        self.use_focal = use_focal
        self.num_classes = num_classes

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        self.softmax = nn.Softmax(dim=1)
        self.early_stopping = EarlyStopping(patience=10, path=save_path)

        # for visualization
        self.train_losses, self.val_losses = [], []
        self.train_macro_f1s, self.val_macro_f1s = [], []

    def compute_class_weights(self, labels_all):
        labels_np = np.array(labels_all)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(self.num_classes), y=labels_np)
        weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        return weights

    def get_loss_fn(self, weights=None):
        if self.use_focal:
            return FocalLoss(alpha=weights, gamma=3.0, reduction='mean')
        else:
            return nn.CrossEntropyLoss(weight=weights)

    def train(self, train_iter, val_iter):
        all_train_labels = []
        for batch in train_iter:
            _, _, _, _, labels, *masks = batch
            labels = torch.stack(labels, dim=0).t()
            all_train_labels.extend(torch.argmax(labels, dim=1).tolist())

        class_weights = self.compute_class_weights(all_train_labels)
        self.loss_fn = self.get_loss_fn(weights=class_weights)

        for epoch in range(self.epoch):
            self.model.train()
            train_preds, train_labels, train_losses = [], [], []

            for batch in tqdm(train_iter, desc=f"Epoch {epoch+1}/{self.epoch}"):
                data, edge_index, _, _, labels, *masks = batch
                labels = torch.stack(labels, dim=0).t()
                data, edge_index, labels = data.to(self.device), edge_index.to(self.device), labels.to(self.device)
                masks = [m.to(self.device) for m in masks]

                self.optimizer.zero_grad()
                outputs = self.model(data, edge_index, *masks) if self.use_mask else self.model(data, edge_index)
                labels = torch.argmax(labels, dim=1)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                pred = torch.argmax(self.softmax(outputs), dim=1)
                train_preds.extend(pred.tolist())
                train_labels.extend(labels.tolist())
                train_losses.append(loss.item())

            val_loss, val_preds, val_labels = self.validate(val_iter)
            print(f"Epoch {epoch+1}/{self.epoch} - Train Loss: {np.mean(train_losses):.4f}, Val Loss: {val_loss:.4f}")
            macro_f1_train = evaluate_model(train_preds, train_labels)[-1]
            macro_f1_val = evaluate_model(val_preds, val_labels)[-1]

            self.train_losses.append(np.mean(train_losses))
            self.val_losses.append(val_loss)
            self.train_macro_f1s.append(macro_f1_train)
            self.val_macro_f1s.append(macro_f1_val)

            self.scheduler.step(val_loss)
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"Early stopped at epoch {epoch+1}")
                break

        self.plot_metrics()

    def validate(self, val_iter):
        self.model.eval()
        val_preds, val_labels, val_losses = [], [], []
        with torch.no_grad():
            for batch in val_iter:
                data, edge_index, _, _, labels, *masks = batch
                labels = torch.stack(labels, dim=0).t()
                data, edge_index, labels = data.to(self.device), edge_index.to(self.device), labels.to(self.device)
                masks = [m.to(self.device) for m in masks]
                outputs = self.model(data, edge_index, *masks) if self.use_mask else self.model(data, edge_index)

                labels = torch.argmax(labels, dim=1)
                loss = self.loss_fn(outputs, labels)

                pred = torch.argmax(self.softmax(outputs), dim=1)
                val_preds.extend(pred.tolist())
                val_labels.extend(labels.tolist())
                val_losses.append(loss.item())

        return np.mean(val_losses), val_preds, val_labels
    
    def test(self, test_iter, model_path):
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for batch in test_iter:
                data, edge_index, _, _, labels, *masks = batch
                labels = torch.stack(labels, dim=0).t()
        
                data, edge_index, labels = data.to(self.device), edge_index.to(self.device), labels.to(self.device)
                masks = [m.to(self.device) for m in masks]
                outputs = self.model(data, edge_index, *masks) if self.use_mask else self.model(data, edge_index)

                pred = torch.argmax(self.softmax(outputs), dim=1)
                test_preds.extend(pred.tolist())
                test_labels.extend(torch.argmax(labels, dim=1).tolist())
                
                # 一轮跳出，测试 attention weights
                # break

        acc, precision, recall, f1, macro_f1 = evaluate_model(test_preds, test_labels)
        
        # special classification report
        report = classification_report(test_labels, test_preds)
        print(report)
        print('---------------------------------')
        cm = confusion_matrix(test_labels, test_preds)
        print(cm)
        # 计算每个类别的 TP、FP、TN、FN
        TP = [cm[i, i] for i in range(cm.shape[0])]
        FP = [sum(cm[:, i]) - cm[i, i] for i in range(cm.shape[0])]
        TN = [sum(cm[i, :]) - cm[i, i] for i in range(cm.shape[0])]
        FN = [sum(cm[i, :]) - cm[i, i] for i in range(cm.shape[0])]

        # 输出每个类别的 TP、FP、TN、FN
        print("\nTP, FP, TN, FN for each class:")
        for i in range(cm.shape[0]):
            print(f"Class {i}: TP={TP[i]}, FP={FP[i]}, TN={TN[i]}, FN={FN[i]}")
        print('---------------------------------')
        print(f"Test Results → Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Macro-F1: {macro_f1:.4f}")

    def plot_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.legend()
        plt.title('Loss over Epochs')

        plt.subplot(1, 2, 2)
        plt.plot(self.train_macro_f1s, label='Train Macro-F1')
        plt.plot(self.val_macro_f1s, label='Val Macro-F1')
        plt.legend()
        plt.title('Macro-F1 over Epochs')

        plt.tight_layout()
        plt.savefig("training_curve.png")
        plt.show()