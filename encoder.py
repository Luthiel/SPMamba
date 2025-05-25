import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from sentence_transformers import SentenceTransformer


class PositionalEncoding(nn.Module):
    """ 位置/时间编码模块 """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x, timestamps=None):
        """
        参数:
            x: 形状为 [batch_size, seq_len, embedding_dim] 的张量
            timestamps: 可选的时间戳，形状为 [batch_size, seq_len]
        """
        if timestamps is not None:
            # 使用实际的时间戳
            batch_size, seq_len = timestamps.shape
            # 将timestamps缩放到合理的范围内
            norm_timestamps = (timestamps % self.pe.size(0)).long()
            
            # 获取对应时间戳的位置编码
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
            time_encodings = self.pe[norm_timestamps[batch_indices, torch.arange(seq_len)]]
            
            x = x + time_encodings
        else:
            # 使用序列位置作为时间编码
            x = x + self.pe[:x.size(1)].unsqueeze(0)
            
        return self.dropout(x)

'''
class ROPE(nn.Module):
    def __init__(self, d_model, dropout=0., max_cache_len=5_000, base=1_000, time_scale=1.0, time_norm=False):
        """
        参数:
            d_model: 输入特征的维度
            max_cache_len: 预计算缓存的最大长度
            base: 基数
            time_scale: 时间缩放因子
        """
        super(ROPE, self).__init__()
        assert d_model % 2 == 0
        self.dim = d_model
        self.base = base
        self.time_scale = time_scale
        self.max_cache_len = max_cache_len
        self.time_norm = time_norm
        self.dropout = nn.Dropout(p=dropout)
        self._register_cache()
    
    def _register_cache(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(self.max_cache_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, timestamps):
        batch_size, seq_len, dim = x.shape
        assert dim == self.dim
        
        t = timestamps.to(dtype=self.inv_freq.dtype)
        if self.time_norm:
            t = (t - t.min()) / (t.max() - t.min() + 1e-8) * self.time_scale
        else:
            t = t - t.min() + 1e-8
            # t = (t - t.min()).clamp(min=1e-8)
        
        if t.max() < self.max_cache_len and t.min() >= 0 and torch.allclose(t, t.long().float()):
            indices = t.long()
            cos = self.cos_cached[indices]
            sin = self.sin_cached[indices]
        else:
            freqs = torch.einsum('bs,d->bsd', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        x_rot = self.dropout(x * cos + self._rotate_half(x) * sin)
        return x_rot
'''

class ROPE(nn.Module):
    def __init__(self, d_model, dropout=0., max_cache_len=5000, base=1000, time_scale=1.0, time_norm=True):
        super(ROPE, self).__init__()
        assert d_model % 2 == 0
        self.dim = d_model
        self.base = base
        self.time_scale = time_scale
        self.max_cache_len = max_cache_len
        self.time_norm = time_norm
        self.dropout = nn.Dropout(p=dropout)
        self._register_cache()

    def _register_cache(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(self.max_cache_len, dtype=inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, timestamps):
        """ x is sentence embeddings with shape [sentence_num, 768] """
        _, dim = x.shape
        assert dim == self.dim, \
            f"Sentence embeddings dimension {dim} doesn't match the expected time embeddings dimension {self.dim}."
        
        t = timestamps.to(self.inv_freq.dtype)
        if self.time_norm:
            t = (t - t.min()).clamp(min=1e-8) / (t.max() - t.min() + 1e-8) * self.time_scale
        else:
            t = (t - t.min()).clamp(min=1e-8)

        if torch.equal(t, t.round()) and t.max() < self.max_cache_len and t.min() >= 0:
            indices = t.long()
            cos = self.cos_cached[indices].squeeze(1)
            sin = self.sin_cached[indices].squeeze(1)
        else:
            freqs = torch.einsum('bs,d->bsd', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).squeeze(1)
            cos = emb.cos()
            sin = emb.sin()

        x_rot = self.dropout(x) * cos + self.dropout(self._rotate_half(x)) * sin
        return x_rot


class MPNet(nn.Module):
    def __init__(self, use_time_enc=False):
        super(MPNet, self).__init__()
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.time_enc = ROPE(d_model=768, dropout=0.1)
        self.use_time_enc = use_time_enc
    
    def forward(self, sentences, timestamps):
        """ 
        Args:
            sentences: list of sentence
        Returns:
            sentence_embd: [batch_size, d_model]
        """
        if len(timestamps.shape) < 2:
            timestamps = timestamps.unsqueeze(1)
        embds = torch.tensor(self.model.encode(sentences))
        # print(f"embds shape: {embds.shape}")
        # print(f"timestamps shape: {timestamps.shape}")
        # todo 消融实验——不对时间编码
        if self.use_time_enc:
            embds_with_time = self.time_enc(embds, timestamps)
        else:
            embds_with_time = embds
        sentence_embd = F.normalize(embds_with_time, p=2, dim=1)
        # print(f"sentence_embd shape: {sentence_embd.shape}")
        return sentence_embd