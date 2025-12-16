# scripts/gnn/feature_extractor.py

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    adj (edges) has shape (B, N, N) with values {0, 1}
    this function normalize the values and return the normalized adj
    formula: \hat{A} = D^{-1/2}(A + I)D^{-1/2}
    """
    B, N, N1 = adj.shape
    assert N == N1, "This formula only support process of NxN tensor with batched dimension"
    eye = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, N, N)
    A = adj.float() + eye
    deg = A.sum(dim=-1) # [B, N]
    inv_sqrt = deg.clamp(min=1e-6).pow(-0.5)
    D1 = inv_sqrt.unsqueeze(-1) # [B,N,1]
    D2 = inv_sqrt.unsqueeze(-2) # [B,1,N]
    return D1 * A * D2

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.ffn = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # x: [B,N,F]
        # A_ht: [B,N,N]
        return self.ffn(torch.bmm(A_hat, x))

class GraphFeaturesExtractor(BaseFeaturesExtractor):
    """
    Input dict:
    ------
    ref: GraphEngine features
    - node_features: (N, F)     float32
    - adj:           (N, N)     int8 / {0,1}
    - node_mask:     (N,)       {0,1}
    ref: ActionMaskWrapper augmented features
    - action_masks:  (N,)       {0, 1}
    - legal_bins:    (3,)       {0,1,2} for [TILE,FUSE,CONVERT]
    - prev_cat_idx:  ()         {0..5}
    """
    def __init__(self, observation_space: spaces.Dict, gcn_hidden: int = 64, gcn_out: int = 64, prev_cat_emb_dim: int = 8):
        # features_dim set to 1 here for base class initialization, the actual value will be override in derived class init
        super().__init__(observation_space, features_dim=1)

        node_features: spaces.Box = observation_space.spaces["node_features"]
        adj: spaces.MultiBinary = observation_space.spaces["adj"]
        node_mask: spaces.MultiBinary = observation_space.spaces["node_mask"]
        assert len(node_features.shape) == 2 and len(adj.shape) == 2 and len(node_mask.shape) == 1, \
            "Expects GraphEngine out dict shape (N,F), (N,N), (N,)"

        self.N, self.F = int(node_features.shape[0]), int(node_features.shape[1])

        # ---- GCN backbone ----
        self.gcn1 = GCNLayer(self.F, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_out)

        # prev_cat embedding (6 CATEGORIES: FUSE/BUFFERIZE/TILE/CONVERT/EVAL/OTHER)
        self.prev_embed = nn.Embedding(6, prev_cat_emb_dim)

        # legal_bins one-hot (each get 3 bins -> 9)
        self.bin_vocab = 3
        self.bin_onehot_dim = 3 * self.bin_vocab

        # final feature dim =  GNN readout(2*gcn_out) + 9 (legal one-hot) + prev_cat_emb_dim
        self._feat_dim = (2 * gcn_out) + self.bin_onehot_dim + prev_cat_emb_dim

        self._features_dim = self._feat_dim

    @property
    def features_dim(self) -> int:
        '''
        override base class init features_dim field.
        '''
        return self._features_dim

    def _readout(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h:    [B,N,D]
        mask: [B,N] bool
        return: [B, 2D]，mean + max（masked）
        """
        m_float = mask.float()
        denom = m_float.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B,1]
        # broadcast mask to dim D, average with denom
        mean_pool = (h * m_float.unsqueeze(-1)).sum(dim=1) / denom  # [B,D]

        # fill -inf to padding position -> avoid being picked as max
        masked_h = h.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_pool = torch.max(masked_h, dim=1).values

        # in case the entire row is masked, use torch.zeros as defualt fallback
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean_pool, max_pool], dim=-1)  # [B,2D]

    def _one_hot_bins(self, bins: torch.Tensor) -> torch.Tensor:
        """
        bins: [B,3] with values 0/1/2
        return: [B, 9]
        """
        B = bins.shape[0]
        oh = F.one_hot(bins.long(), num_classes=self.bin_vocab)  # [B,3,3]
        return oh.view(B, -1).float()  # [B,9]

    def forward(self, obs: dict) -> torch.Tensor:
        # if no batch provided, default B=1
        def make_batch(x: torch.Tensor, want_dim: int) -> torch.Tensor:
            if x.dim() == want_dim - 1:
                return x.unsqueeze(0)
            return x
        # [B,N,F]
        node_feats = make_batch(obs["node_features"], 3)
        # [B,N,N]
        adj = make_batch(obs["adj"], 3)
        # [B,N]
        node_mask = make_batch(obs["node_mask"], 2).bool()
        # [B,3]
        # action mask wrapper -> multi-discrete -> (1, 9)
        bins = make_batch(obs["legal_bins"], 2)
        # [B]
        # action mask wrapper -> Discrete -> (1, 6)
        prev = make_batch(obs["prev_cat_idx"], 1)
        prev_idx = prev.argmax(dim=-1).long().view(-1)

        # ---- GCN ----
        # [B,N,N]
        A_hat = _normalize_adj(adj)
        # [B,N,N]
        h = F.relu(self.gcn1(node_feats, A_hat))
        # [B,N,D], where D=N in this case
        h = F.relu(self.gcn2(h, A_hat))

        g = self._readout(h, node_mask) # [B, 2D]
        b = bins.float() # [B, 9]
        e = self.prev_embed(prev_idx).squeeze(1) # [B, E]

        return torch.cat([g, b, e], dim=-1) # [B, features_dim]
