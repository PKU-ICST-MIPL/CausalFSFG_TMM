# Bi-FRN-main/models/causal_modules.py (新建文件)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class IMSE_Residual(nn.Module):
    """
    Improved IMSE with Residual Modulation.
    Instead of aggressively reducing the feature dimension of the main path,
    we only reduce dimension for the 'causal branch' to calculate weights,
    and then apply weights back to the original high-dim features via residual connection.
    """
    def __init__(self, in_channels_list, reduce_dim=64, num_heads=4):
        super().__init__()
        self.num_scales = len(in_channels_list)
        
        self.reduce_layers = nn.ModuleList([
            nn.Conv2d(c, reduce_dim, 1) for c in in_channels_list
        ])
        
        self.interventional_tokens = nn.Parameter(torch.randn(self.num_scales, 1, reduce_dim))
        
        self.inter_scale_transformer = nn.TransformerEncoderLayer(
            d_model=reduce_dim, 
            nhead=num_heads, 
            dim_feedforward=reduce_dim * 2,
            batch_first=True,
            dropout=0.1
        )

        self.modulate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(reduce_dim, c),
                nn.Tanh()
            ) for c in in_channels_list
        ])

        self.out_channel = in_channels_list[-1]
        self.align_layers = nn.ModuleList([
            nn.Conv2d(c, self.out_channel, 1) for c in in_channels_list
        ])
        
        self.pool = self.pool = nn.MaxPool2d(2)

    def forward(self, multi_scale_features):
        batch_size = multi_scale_features[0].size(0)
        
        scale_summary_vectors = []
        for i in range(self.num_scales):
            reduced = self.reduce_layers[i](multi_scale_features[i])
            summary = F.adaptive_avg_pool2d(reduced, (1, 1)).squeeze(-1).squeeze(-1)
            scale_summary_vectors.append(summary)

        scale_sequence = torch.stack(scale_summary_vectors, dim=1) # (B, 4, C_small)
        tokens_expanded = self.interventional_tokens.expand(self.num_scales, batch_size, -1).transpose(0, 1)
        combined_sequence = torch.cat([scale_sequence, tokens_expanded], dim=1)
        
        contextualized_sequence = self.inter_scale_transformer(combined_sequence)
        
        feature_context = contextualized_sequence[:, :self.num_scales, :] # (B, 4, C_small)

        modulated_features = []
        
        for i in range(self.num_scales):
            context = feature_context[:, i, :]
            modulation = self.modulate_layers[i](context).unsqueeze(-1).unsqueeze(-1)
            
            feat_enhanced = multi_scale_features[i] * (1 + modulation)
            
            feat_aligned = self.align_layers[i](feat_enhanced)
            modulated_features.append(feat_aligned)

        f1_pooled = self.pool(modulated_features[0])
        f2_pooled = self.pool(modulated_features[1] + f1_pooled)
        f3_pooled = self.pool(modulated_features[2] + f2_pooled)
        intervened_feature = modulated_features[3] + f3_pooled
        
        return intervened_feature


class IMFR_Head(nn.Module):
    """
    A simplified IMFR module that acts as a replacement for FMRM.
    It takes support and query features and returns negative L2 distance logits.
    """
    def __init__(self, in_channels, k):
        super().__init__()
        self.k = k
        self.in_channels = in_channels

        # --- Query Enhancement Part ---
        # Note: To simplify, we make the mask generator have a static structure.
        # This part is simplified from the original paper for minimal code change.
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # --- Reconstruction Part (Cross-Attention) ---
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)

    def forward(self, support_features, query_features):
        """
        Args:
            support_features (Tensor): Support features, shape (way, shot, C, H, W)
            query_features (Tensor): Query features, shape (num_query, C, H, W)
        Returns:
            Tensor: Negative L2 distance logits, shape (num_query, way)
        """
        way, shot, C, H, W = support_features.shape
        num_query = query_features.shape[0]

        # 1. Query Enhancement
        mask = self.mask_generator(query_features)
        mask_flat = mask.view(num_query, -1)
        topk_vals, _ = torch.topk(mask_flat, self.k, dim=1)
        binary_mask = (mask >= topk_vals[:, -1].view(num_query, 1, 1, 1)).float()
        enhanced_query_features = query_features + query_features * binary_mask

        # 2. Create Prototypes
        support_prototypes = support_features.mean(dim=1)

        # 3. Reconstruction
        q_flat = enhanced_query_features.flatten(2).transpose(1, 2)
        proto_flat = support_prototypes.flatten(2).transpose(1, 2)
        
        q_proj = self.q_proj(q_flat).unsqueeze(1)
        k_proj = self.k_proj(proto_flat).unsqueeze(0)
        v_proj = self.v_proj(proto_flat).unsqueeze(0)

        attn_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(C)
        attn_probs = F.softmax(attn_scores, dim=-1)
        reconstructed_q_flat = torch.matmul(attn_probs, v_proj)

        # 4. Distance Calculation
        # Calculate negative squared Euclidean distance between reconstructed query and prototype values
        distances = -torch.sum((reconstructed_q_flat - v_proj)**2, dim=[-1, -2])

        return distances