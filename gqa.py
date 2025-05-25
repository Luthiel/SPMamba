import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn
from typing import Optional, Tuple, Union
import numpy as np

class GQA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        query_heads: int,
        kv_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(GQA, self).__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if self.query_heads % self.kv_heads != 0:
            raise ValueError(
                f"query_heads ({query_heads}) must be divisible by kv_heads ({kv_heads})"
            )
        elif (embed_dim % self.query_heads != 0) or (embed_dim % self.kv_heads != 0):
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by query_heads ({query_heads}) and kv_heads ({kv_heads})"
            )

        head_dim = embed_dim // query_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )
        if not head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128"
            )
        
        # query, key, value projections
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = nn.Linear(
            embed_dim, kv_embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, kv_embed_dim, bias=bias, device=device, dtype=dtype
        )
        
        # normalization
        self.norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.norm = nn.LayerNorm(
                embed_dim, eps=layer_norm_eps, device=device, dtype=dtype)

        # output projection
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # initialize weights and biases
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def scaled_dot_product_gqa(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        adj: Tensor,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False
    ):
        if (mask is not None) and (is_causal is not None):
            raise ValueError(
                "Only one of 'mask' and 'is_causal' should be provided, but got both."
            )
        elif not query.ndim == key.ndim == value.ndim == 4:
            raise ValueError(
                f"Expected query, key, and value to be 4-dimensional, but got shapes "
                f"{query.shape}, {key.shape}, and {value.shape}."
            )

        # Move sequence length dimension to axis 2.
        # This makes the attention operations below *much* faster.
        query = rearrange(query, "b n h d -> b h n d")
        key = rearrange(key, "b s h d -> b h s d") # n = s
        value = rearrange(value, "b s h d -> b h s d")

        bq, hq, nq, dq = query.shape
        bk, hk, nk, dk = key.shape
        bv, hv, nv, dv = value.shape
        if not (bq == bk == bv and dq == dk == dv):
            raise ValueError(
                "Expected query, key, and value to have the same batch size (dim=0) and "
                f"embedding dimension (dim=3), but got query: {query.shape}, "
                f"key: {key.shape}, and value: {value.shape}."
            )
        elif (hk != hv) or (nk != nv):
            raise ValueError(
                "Expected key and value to have the same size in dimensions 1 and 2, but "
                f"got key: {key.shape} and value: {value.shape}."
            )
        elif hq % hk != 0:
            raise ValueError(
                "Expected query heads to be a multiple of key/value heads, but got "
                f"query: {query.shape} and key/value: {key.shape}."
            )

        if scale is None:
            scale = query.size(-1) ** 0.5
        query = query / scale

        num_head_groups = hq // hk
        query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
        similarity = einsum(query, key, "b g h n d, b h s d -> b g h n s")

        if is_causal:
            mask = torch.ones((bq, nq, nk), device=query.device, dtype=torch.bool).tril_()

        if mask is not None:
            if mask.ndim == 2:
                mask = rearrange(mask, "b s -> b () () () s") # 一维掩码
            elif mask.ndim == 3:
                mask = rearrange(mask, "b n s -> b () () n s") # 二维掩码
            similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)

        if is_causal:
            neighbor_mask = rearrange(adj, "b n s -> b () () n s").bool().to(query.device)
            similarity.masked_fill_(~neighbor_mask, torch.finfo(similarity.dtype).min)
            attention = F.softmax(similarity, dim=-1) # causal attention and aggregate meesage from neighbors
        else:
            attention = F.softmax(similarity, dim=-1)
            
        if dropout > 0.:
            attention = F.dropout(attention, p=dropout)

        out = einsum(attention, value, "b g h n s, b h s d -> b g h n d")
        out = rearrange(out, "b g h n d -> b n (h g) d")

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = rearrange(attention, "b g h n s -> b n s (h g)")
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)

        return out, attn_weights

    '''
    def softmax(self, x, adj, dim=(-1, -2)):
        # adj = rearrange(adj, "b n n -> b () () n n")
        x = x - torch.ones_like(x) * torch.max(x, dim=dim, keepdim=True)
        
        exps = torch.exp(x)
        sum_exps = torch.sum(torch.exp(x * adj), dim=dim) # [batch_size, heads, groups, 1]
        
        softmax_values = exps / sum_exps
        return softmax_values
    '''

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        adj: Tensor,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = None,
        average_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)
        # print(f"Adj shape: {adj.shape}")

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.query_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.kv_heads)
        # Apply attention, then fold 'h' attention heads back into 'd'.
        x, attn = self.scaled_dot_product_gqa(
            query=q,
            key=k,
            value=v,
            adj=adj,
            mask=attn_mask,
            is_causal=is_causal,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        x = rearrange(x, "b n h d -> b n (h d)")
        if self.layer_norm:
            assert self.norm is not None
            x = self.norm(x)
        
        x = self.out_proj(x)

        return x, attn

class DGSL(nn.Module):
    def __init__(self, in_channels,  hidden_channels, out_channels, query_heads, kv_heads, **kwargs):
        super(DGSL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn = GQA(in_channels, query_heads, kv_heads, **kwargs) # 128, 8, 4, 0.2, **kwargs
        self.weight = nn.Parameter(torch.ones(3))
    
    def forward(self, x, edge_index, mask1=None, mask2=None):
        b, n, d = x.shape
        query, key, value = x, x, x
        adj = torch.zeros(b, n, n)
        for i in range(b):
            adj[i][edge_index[i][0], edge_index[i][1]] = 1
        
        global_attn, g_aw = self.attn(query, key, value, adj, is_causal=True, need_weights=True) # 
        
        # for attn weight visualization
        causal_aw = np.array(g_aw.cpu().detach().numpy())
        np.savez('causal_aw.npz', causal_aw)
        
        if mask1 is not None and mask2 is not None:
            local_attn_1, aw_1 = self.attn(query, key, value, adj, attn_mask=mask1.bool(), need_weights=True) # , need_weights=True
            local_attn_2, aw_2 = self.attn(query, key, value, adj, attn_mask=mask2.bool(), need_weights=True) # , need_weights=True
            local_attns = [local_attn_1, local_attn_2]
            
            # for attn weight visualization
            sub_top_aw = np.array(aw_1.cpu().detach().numpy())
            trail_aw = np.array(aw_2.cpu().detach().numpy())
            
            np.savez('sub_top_aw.npz', sub_top_aw)
            np.savez('trail_aw.npz', trail_aw)
        else:
            return global_attn
        
        attns = torch.stack([global_attn] + local_attns, dim=-1) # [b, n, d, 3]
        
        w = F.softmax(self.weight, dim=0)  
        out = (attns * w).sum(dim=-1)
        
        return out