import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class IMFR(nn.Module):
    """
    Feature Mutual Reconstruction Module
    """
    def __init__(self, hidden_size, inner_size=None, num_patch=25, drop_prob=0., k=3):
        super(IMFR, self).__init__()

        self.hidden_size = hidden_size
        self.inner_size = inner_size if inner_size is not None else hidden_size//8
        self.num_patch = num_patch

        self.k = k

        dim_per_head = inner_size
        self.num_heads = 1
        inner_dim = self.inner_size * self.num_heads
        self.to_qkv = nn.Sequential(
            nn.Linear(self.hidden_size, inner_dim * 3, bias=False),
            )

        self.dropout = nn.Dropout(drop_prob)

        self.mask_generator = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size // 4, kernel_size=1),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(),
            nn.Conv2d(hidden_size // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.w_full = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.w_masked = nn.Parameter(torch.FloatTensor([0.01]), requires_grad=True)


    def compute_distances(self, query_a, key_a, value_a, query_b, key_b, value_b, features_a, features_b):

        # 1) feature reconstruction
        value_a = value_a.unsqueeze(0)
        value_b = value_b.unsqueeze(1)

        n_way = value_a.size(1)
        n_query = value_b.size(0)
        s_patch = value_a.size(3)

        # Reconstructed features B
        att_scores = torch.matmul(query_b.unsqueeze(1), key_a.unsqueeze(0).transpose(-1, -2).contiguous())
        att_probs = nn.Softmax(dim=-1)(att_scores / math.sqrt(self.inner_size))
        att_probs = self.dropout(att_probs)
        # (N_query x N_way x 1 x HW x N-shot*HW) x (1 x N_way x 1 x N-shot*HW x C) -> (N_query x N_way x 1 x HW x C)
        reconstructed_features_b = torch.matmul(att_probs, value_a)

        # Reconstructed features A
        att_scores = torch.matmul(query_a.unsqueeze(0), key_b.unsqueeze(1).transpose(-1, -2).contiguous())
        att_probs = nn.Softmax(dim=-1)(att_scores / math.sqrt(self.inner_size))
        att_probs = self.dropout(att_probs)

        # (N_query x N_way x 1 x N-shot*HW x HW) x (N_query x 1 x 1 x HW x C) -> (N_query x N_way x 1 x N-shot*HW x C)
        reconstructed_features_a = torch.matmul(att_probs, value_b)

        assert reconstructed_features_a.size(-1) == self.inner_size
        assert reconstructed_features_b.size(-1) == self.inner_size
        assert value_a.size(-1) == self.inner_size
        assert value_b.size(-1) == self.inner_size

        # 2) compute the Euclide distance
        sq_similarity = -torch.sum((value_a.view(value_a.size(0), value_a.size(1), -1)-reconstructed_features_a.view(reconstructed_features_a.size(0), reconstructed_features_a.size(1), -1))**2, dim=-1)
        qs_similarity = -torch.sum((value_b.view(value_b.size(0), value_b.size(1), -1)-reconstructed_features_b.view(reconstructed_features_b.size(0), reconstructed_features_b.size(1), -1))**2, dim=-1)

        return sq_similarity, qs_similarity

    def forward(self, features_a, features_b):
        # projection of features a
        way, dim, shot, h, w = features_a.shape
        mask_a = self.mask_generator(features_a.permute(0,2,1,3,4).flatten(0,1)).view(way, shot, 1, h, w).permute(0,2,1,3,4).contiguous()
        mask_a_flat = mask_a.flatten(-2)
        _, topk_indices_a = torch.topk(mask_a_flat, self.k, dim=-1)
        expanded_indices_a = topk_indices_a.expand(-1, dim, -1, -1)
        topk_features_a = torch.gather(features_a.flatten(-2), dim=-1, index=expanded_indices_a).flatten(-2).permute(0, 2, 1).contiguous()

        features_a = features_a.view(features_a.size(0), features_a.size(1), -1).permute(0, 2, 1).contiguous()

        b_a, l_a, d_a = features_a.shape
        _, k_a, _ = topk_features_a.shape
        features_a = torch.cat([features_a, topk_features_a], dim=1)

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)
        qkv_a = self.to_qkv(features_a)
        # (3,b,num_heads,l,dim_per_head)
        qkv_a = qkv_a.view(b_a, l_a+k_a, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        query_a, key_a, value_a = qkv_a.chunk(3)
        query_a, key_a, value_a = query_a.squeeze(0), key_a.squeeze(0), value_a.squeeze(0)



        # projection of features b
        mask_b = self.mask_generator(features_b)
        mask_b_flat = mask_b.flatten(-2)
        _, topk_indices_b = torch.topk(mask_b_flat, self.k, dim=-1)
        expanded_indices_b = topk_indices_b.expand(-1, dim, -1)
        topk_features_b = torch.gather(features_b.flatten(-2), dim=-1, index=expanded_indices_b).permute(0, 2, 1).contiguous()
        features_b = features_b.view(features_b.size(0), features_b.size(1), -1).permute(0, 2, 1).contiguous()

        b_b, l_b, d_b = features_b.shape
        _, k_b, _ = topk_features_b.shape
        features_b = torch.cat([features_b, topk_features_b], dim=1)

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)
        qkv_b = self.to_qkv(features_b)
        # (3,b,num_heads,l,dim_per_head)
        qkv_b = qkv_b.view(b_b, l_b+k_b, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        query_b, key_b, value_b = qkv_b.chunk(3)
        query_b, key_b, value_b = query_b.squeeze(0), key_b.squeeze(0), value_b.squeeze(0)

        # compute the total spatial similarity
        distances = self.compute_distances(query_a, key_a, value_a, query_b, key_b, value_b, features_a, features_b)

        return distances