import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ABM(nn.Module):
    """
    Attention-based Matching Module
    """
    def __init__(self, feature_dim=512, seq_len=5000):
        super(ABM, self).__init__()
        self.fc_A = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        self.fc_B = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(seq_len * 2, seq_len // 5),
            nn.ReLU(),
            nn.Linear(seq_len // 5, 2 * seq_len),
        )
        self.sigmoid = nn.Sigmoid()


    def cal_simility(self, A, B):
        """
        Returns:
            torch.Tensor: The cosine similarity matrix. Shape: [1, 1, N].
        """
        A = F.normalize(A, p=2, dim=1)  
        B = F.normalize(B, p=2, dim=1)  
        cosine_similarity = torch.sum(A * B, dim=1, keepdim=True)  # shape: [1, 1, N]

        return cosine_similarity

    def forward(self, A, B, opt):
        """
        Returns:
            torch.Tensor: Fused tensor. Shape: [1, 512, 5000].
        """
        A_proj = F.relu(self.fc_A(A))  # shape: [1, 512, 900]
        B_proj = F.relu(self.fc_B(B))  # shape: [1, 512, 900]

        sim_mat = self.sigmoid(self.cal_simility(A_proj, B_proj))
        
        if opt == 'inhibit':
            weighted_A = (1 - sim_mat) * A_proj  # shape: [1, 512, 900]
            weighted_B = (1 - sim_mat) * B_proj  # shape: [1, 512, 900]
        else:
            weighted_A = sim_mat * A_proj  # shape: [1, 512, 900]
            weighted_B = sim_mat * B_proj  # shape: [1, 512, 900]

        combined = torch.cat((weighted_A, weighted_B), dim=2)  # shape: [1, 1024, 900]
        fused_tensor = F.relu(self.fc_fusion(combined))  # shape: [1, 512, 900]

        return fused_tensor

class FAM(nn.Module):
    def __init__(self, feature_dim=512, N=900):
        super(FAM, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.matching = ABM(feature_dim, N)
        self.adapt_pooling = nn.AdaptiveAvgPool1d(N)

    def forward(self, vision_feat, text_feat, graph_feat):
        """
        Args:
            feats (torch.Tensor): Modality features. Shape: [batch_size, seq_len, feat_dim].
        Returns:
            tuple: A tuple containing the fused vision, text, and graph features.
        """
        vision_feat = [[self.adapt_pooling(feat) for feat in batch] for batch in vision_feat]
        text_feat = [[self.adapt_pooling(feat) for feat in batch] for batch in text_feat]
        graph_feat = [[self.adapt_pooling(feat) for feat in batch] for batch in graph_feat]

        fused_vt = self.matching(vision_feat, text_feat, 'inhibit')
        fused_vg = self.matching(vision_feat, graph_feat, 'inhibit')
        fused_tg = self.matching(text_feat, graph_feat, 'None')

        return fused_vt, fused_vg, fused_tg

    def reshape_to_square(self, tensor):
        """
        Reshapes a tensor to a square shape.
        """
        B, C, N = tensor.shape # [batch_size, channel/feat_dim, seq_len]
        side_length = int(np.ceil(np.sqrt(N)))
        padded_length = side_length ** 2
        
        padded_tensor = torch.zeros((B, C, padded_length), device=tensor.device)
        padded_tensor[:, :, :N] = tensor

        square_tensor = padded_tensor.view(B, C, side_length, side_length)
        
        return square_tensor, side_length, side_length, N
    
    
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q_feat, K_feat):

        B, C, N = Q_feat.shape

        Q_feat = Q_feat.permute(0, 2, 1)
        K_feat = K_feat.permute(0, 2, 1)


        Q = self.q_proj(Q_feat) 
        K = self.k_proj(K_feat) 
        V = self.v_proj(K_feat) 

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(C, dtype=torch.float32))
        attn_weights = self.softmax(attn_scores)  # shape: [B, N, N]

        attn_feats = torch.matmul(attn_weights, V)  # shape: [B, N, C]
        attn_feats = attn_feats.permute(0, 2, 1)

        return attn_feats

class Fusion(nn.Module): # Attention-based Feature Fusion Module
    def __init__(self, feat_dim):
        super(Fusion, self).__init__()
        self.CA1 = CrossAttention(feat_dim)
        self.CA2 = CrossAttention(feat_dim)
        self.relu = nn.ReLU()
    
    def forward(self, vision, text, graph):
        vt = self.CA1(vision, text)
        tg = self.CA2(text, graph)
        fused = self.relu(vt + text + tg)
        return fused