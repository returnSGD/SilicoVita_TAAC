#!/usr/bin/env python3
"""
Enhanced Recommender: Multi-View Heterogeneous Transformer for CVR Prediction
TAAC2026 Competition — Improved Architecture (v2)

Key improvements over baseline (AUC ~0.65):
  1. Larger capacity  d_model=128, n_layers=4, n_heads=8, d_ff=512
  2. Larger vocab     32768 (reduces hash collision)
  3. Multi-view readout  target / user / item mean-pool (vs. only target token)
  4. DIN-style attention  target-aware pooling over behavioral sequences
  5. Explicit feature cross  user ⊙ item Hadamard product appended to head input
  6. Richer sequence parsing  use ALL feature structs, not just the first
  7. Focal BCE loss  handles positive/negative class imbalance
  8. Label smoothing  reduces over-confidence
  9. Stochastic depth  improves regularization in deeper stack
 10. Cosine LR with longer warm-up

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Token Embedding  (FeatIDEmb ⊙ ValueGate + TypeEmb + PosEmb)     │
  ├──────────────────────────────────────────────────────────────────┤
  │  4 × HeterogeneousTransformerLayer  (Pre-LN, HetBias, GELU FFN)  │
  │      + StochasticDepth per layer                                 │
  ├──────────────────────────────────────────────────────────────────┤
  │  Multi-View Readout                                              │
  │    target_repr  = x[:, 0]                                        │
  │    user_repr    = masked mean-pool over TT_USER tokens           │
  │    item_repr    = masked mean-pool over TT_ITEM tokens           │
  │    din_repr     = DIN attention(target, seq tokens)              │
  ├──────────────────────────────────────────────────────────────────┤
  │  Interaction:  concat + user ⊙ item cross product                │
  ├──────────────────────────────────────────────────────────────────┤
  │  Deep MLP head  → CVR logit                                      │
  └──────────────────────────────────────────────────────────────────┘
"""

import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import partial

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)


# ════════════════════════════════════════════════════════════
# 0. Token Type Constants & Configuration
# ════════════════════════════════════════════════════════════

TT_TARGET   = 0
TT_USER     = 1
TT_ITEM     = 2
TT_ACT_SEQ  = 3
TT_CONT_SEQ = 4
TT_ITEM_SEQ = 5
N_TT        = 6
SEQ_TYPES   = {TT_ACT_SEQ, TT_CONT_SEQ, TT_ITEM_SEQ}


@dataclass
class Config:
    # ── Embedding ──────────────────────────────────────────
    d_model: int      = 128     # ↑ from 64
    vocab_size: int   = 32768   # ↑ from 4096  (fewer hash collisions)
    max_seq_len: int  = 50
    max_field_t: int  = 48

    # ── Transformer ────────────────────────────────────────
    n_layers: int  = 4          # ↑ from 3
    n_heads: int   = 8          # ↑ from 4
    d_ff: int      = 512        # ↑ from 256
    dropout: float = 0.15
    drop_path: float = 0.10     # NEW: stochastic depth probability

    # ── Training ───────────────────────────────────────────
    batch_size: int   = 256
    lr: float         = 3e-4
    wd: float         = 1e-4
    epochs: int       = 60
    grad_clip: float  = 1.0
    warmup_epochs: int = 4      # ↑ from 2

    # ── Loss ───────────────────────────────────────────────
    focal_gamma: float = 2.0    # NEW: focal loss gamma (0 = vanilla BCE)
    label_smooth: float = 0.05  # NEW: label smoothing

    # ── Label ──────────────────────────────────────────────
    cvr_type: int = 1


# ════════════════════════════════════════════════════════════
# 1. Data Parsing
# ════════════════════════════════════════════════════════════

def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_val(v: float) -> float:
    v = float(v)
    return math.log1p(abs(v)) * (1.0 if v >= 0 else -1.0)


def parse_label(label_arr, cvr_type: int = 1) -> int:
    if label_arr is None:
        return 0
    for a in label_arr:
        if _get(a, "action_type", 0) >= cvr_type:
            return 1
    return 0


def parse_feature_array(
    feat_arr, max_tokens: int, vocab_size: int = 32768
) -> Tuple[List[int], List[float]]:
    fids: List[int]    = []
    fvals: List[float] = []
    if feat_arr is None:
        return fids, fvals

    for feat in feat_arr:
        fid   = int(_get(feat, "feature_id", 0)) % (vocab_size - 1) + 1
        ftype = _get(feat, "feature_value_type", "") or ""

        def emit(v):
            fids.append(fid)
            fvals.append(normalize_val(v))

        if "int_value" in ftype and "array" not in ftype:
            v = _get(feat, "int_value", None)
            if v is not None:
                emit(v)
        if ftype == "float_value":
            v = _get(feat, "float_value", None)
            if v is not None:
                emit(v)
        if "int_array" in ftype:
            arr = _get(feat, "int_array", None)
            if arr is not None:
                for x in arr[:8]:
                    emit(x)
        if "float_array" in ftype:
            arr = _get(feat, "float_array", None)
            if arr is not None:
                for x in arr[:8]:
                    emit(x)
        if len(fids) >= max_tokens:
            break

    return fids[:max_tokens], fvals[:max_tokens]


def parse_seq_sub(
    seq_feature, sub_key: str, max_seq_len: int, vocab_size: int = 32768
) -> Tuple[List[int], List[int]]:
    """
    Improved: uses ALL feature structs (not just the first), interleaving
    tokens from each struct with a shared position index.
    """
    if seq_feature is None:
        return [], []
    sub = _get(seq_feature, sub_key, None)
    if sub is None or len(sub) == 0:
        return [], []

    tok_ids: List[int]  = []
    positions: List[int] = []

    for feat in sub:
        arr = _get(feat, "int_array", None)
        if arr is not None:
            for pos, v in enumerate(arr[:max_seq_len]):
                tok_ids.append(int(v) % (vocab_size - 1) + 1)
                positions.append(pos + 1)
        if len(tok_ids) >= max_seq_len:
            break

    return tok_ids[:max_seq_len], positions[:max_seq_len]


# ════════════════════════════════════════════════════════════
# 2. Dataset & Collate
# ════════════════════════════════════════════════════════════

class TAADataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.cfg = cfg
        self.samples: List[Dict] = []
        print(f"  Tokenizing {len(df)} samples...", flush=True)
        for _, row in df.iterrows():
            self.samples.append(self._process(row))
        print(f"  Done. Avg tokens/sample: "
              f"{np.mean([self._count(s) for s in self.samples]):.1f}")

    @staticmethod
    def _count(s: Dict) -> int:
        return (1 + len(s["u_fids"]) + len(s["i_fids"])
                + len(s["act_ids"]) + len(s["cnt_ids"]) + len(s["itm_ids"]))

    def _process(self, row) -> Dict:
        cfg = self.cfg
        label     = parse_label(row["label"], cfg.cvr_type)
        target_id = int(row["item_id"]) % (cfg.vocab_size - 1) + 1

        u_fids, u_fvals = parse_feature_array(row["user_feature"], cfg.max_field_t, cfg.vocab_size)
        i_fids, i_fvals = parse_feature_array(row["item_feature"], cfg.max_field_t, cfg.vocab_size)

        seq = row["seq_feature"]
        act_ids, act_pos = parse_seq_sub(seq, "action_seq",  cfg.max_seq_len, cfg.vocab_size)
        cnt_ids, cnt_pos = parse_seq_sub(seq, "content_seq", cfg.max_seq_len, cfg.vocab_size)
        itm_ids, itm_pos = parse_seq_sub(seq, "item_seq",    cfg.max_seq_len, cfg.vocab_size)

        return dict(
            label=label, target_id=target_id,
            u_fids=u_fids, u_fvals=u_fvals,
            i_fids=i_fids, i_fvals=i_fvals,
            act_ids=act_ids, act_pos=act_pos,
            cnt_ids=cnt_ids, cnt_pos=cnt_pos,
            itm_ids=itm_ids, itm_pos=itm_pos,
        )

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def collate_fn(batch: List[Dict], cfg: Config) -> Dict:
    B = len(batch)
    all_ids, all_vals, all_types, all_pos = [], [], [], []

    for s in batch:
        ids, vals, types, pos = [], [], [], []

        def add(i, v, t, p):
            ids.append(i); vals.append(v); types.append(t); pos.append(p)

        add(s["target_id"], 1.0, TT_TARGET, 0)
        for fid, fv in zip(s["u_fids"], s["u_fvals"]):
            add(fid, fv, TT_USER, 0)
        for fid, fv in zip(s["i_fids"], s["i_fvals"]):
            add(fid, fv, TT_ITEM, 0)
        for tid, p in zip(s["act_ids"], s["act_pos"]):
            add(tid, 1.0, TT_ACT_SEQ, p)
        for tid, p in zip(s["cnt_ids"], s["cnt_pos"]):
            add(tid, 1.0, TT_CONT_SEQ, p)
        for tid, p in zip(s["itm_ids"], s["itm_pos"]):
            add(tid, 1.0, TT_ITEM_SEQ, p)

        all_ids.append(ids); all_vals.append(vals)
        all_types.append(types); all_pos.append(pos)

    T = max(len(x) for x in all_ids)
    t_ids   = torch.zeros(B, T, dtype=torch.long)
    t_vals  = torch.zeros(B, T, dtype=torch.float)
    t_types = torch.zeros(B, T, dtype=torch.long)
    t_pos   = torch.zeros(B, T, dtype=torch.long)
    t_mask  = torch.zeros(B, T, dtype=torch.bool)

    for i in range(B):
        L = len(all_ids[i])
        t_ids  [i, :L] = torch.tensor(all_ids  [i], dtype=torch.long)
        t_vals [i, :L] = torch.tensor(all_vals [i], dtype=torch.float)
        t_types[i, :L] = torch.tensor(all_types[i], dtype=torch.long)
        t_pos  [i, :L] = torch.tensor(all_pos  [i], dtype=torch.long)
        t_mask [i, :L] = True

    labels = torch.tensor([s["label"] for s in batch], dtype=torch.float)
    return dict(ids=t_ids, vals=t_vals, types=t_types,
                pos=t_pos, mask=t_mask, labels=labels)


# ════════════════════════════════════════════════════════════
# 3. Model Components
# ════════════════════════════════════════════════════════════

class TokenEmbedding(nn.Module):
    """
    Unified Feature-as-Token embedding (same design as baseline,
    but with larger d_model and vocab_size).
    """
    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        self.feat_emb = nn.Embedding(cfg.vocab_size, D, padding_idx=0)
        self.type_emb = nn.Embedding(N_TT, D)
        self.pos_emb  = nn.Embedding(cfg.max_seq_len + 2, D)
        self.val_gate = nn.Sequential(
            nn.Linear(1, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D), nn.Sigmoid(),
        )
        self.norm    = nn.LayerNorm(D)
        self.dropout = nn.Dropout(cfg.dropout)
        nn.init.normal_(self.feat_emb.weight, std=0.01)
        nn.init.normal_(self.type_emb.weight, std=0.01)
        nn.init.normal_(self.pos_emb.weight,  std=0.01)

    def forward(self, ids, vals, types, pos):
        pos  = pos.clamp(0, self.pos_emb.num_embeddings - 1)
        gate = self.val_gate(vals.unsqueeze(-1))
        x    = self.feat_emb(ids) * gate
        x    = x + self.type_emb(types)
        x    = x + self.pos_emb(pos)
        return self.dropout(self.norm(x))


class HetBias(nn.Module):
    """Learnable n_heads × N_TT × N_TT attention bias."""
    def __init__(self, n_heads: int):
        super().__init__()
        self.H    = n_heads
        self.bias = nn.Parameter(torch.zeros(n_heads, N_TT, N_TT))

    def forward(self, tq, tk):
        B, Tq = tq.shape; Tk = tk.shape[1]; H = self.H
        tq_bh = tq.unsqueeze(1).expand(B, H, Tq).reshape(B * H, Tq)
        tk_bh = tk.unsqueeze(1).expand(B, H, Tk).reshape(B * H, Tk)
        bias  = self.bias.unsqueeze(0).expand(B, H, N_TT, N_TT).reshape(B * H, N_TT, N_TT)
        row_idx = tq_bh.unsqueeze(-1).expand(B * H, Tq, N_TT)
        rows    = bias.gather(1, row_idx)
        col_idx = tk_bh.unsqueeze(1).expand(B * H, Tq, Tk)
        result  = rows.gather(2, col_idx)
        return result.reshape(B, H, Tq, Tk)


class StochasticDepth(nn.Module):
    """Drop entire residual branch with probability p during training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep) / keep
        return x * noise


class HetTransformerLayer(nn.Module):
    """
    Transformer layer with HetBias + StochasticDepth (Pre-LayerNorm).
    """
    def __init__(self, cfg: Config, drop_path_prob: float = 0.0):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0
        self.H  = H
        self.Dh = D // H

        self.qkv  = nn.Linear(D, 3 * D, bias=False)
        self.out  = nn.Linear(D, D)
        self.het  = HetBias(H)
        self.ffn  = nn.Sequential(
            nn.Linear(D, cfg.d_ff), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, D),
        )
        self.n1  = nn.LayerNorm(D)
        self.n2  = nn.LayerNorm(D)
        self.dp  = nn.Dropout(cfg.dropout)
        self.sd  = StochasticDepth(drop_path_prob)   # NEW

    def forward(self, x, types, mask):
        B, T, D = x.shape
        H, Dh   = self.H, self.Dh

        # ── Multi-Head Self-Attention ────────────────────────
        residual = x
        x = self.n1(x)
        Q, K, V = self.qkv(x).split(D, dim=-1)
        def sh(t): return t.view(B, T, H, Dh).transpose(1, 2)
        Q, K, V = sh(Q), sh(K), sh(V)
        logits  = (Q @ K.transpose(-2, -1)) / math.sqrt(Dh)
        logits += self.het(types, types)
        if mask is not None:
            logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
        w = F.softmax(logits, dim=-1)
        w = torch.nan_to_num(w)
        w = self.dp(w)
        attn = (w @ V).transpose(1, 2).reshape(B, T, D)
        x = residual + self.sd(self.dp(self.out(attn)))   # stochastic depth

        # ── Feed-Forward ─────────────────────────────────────
        x = x + self.sd(self.dp(self.ffn(self.n2(x))))
        return x


class DINAttention(nn.Module):
    """
    Target-aware sequence attention (simplified DIN).
    Computes attention weights between target repr and each seq token,
    then returns a weighted sum.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Project concat[target, seq_tok] → scalar score
        self.score = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        target: torch.Tensor,  # (B, D)
        seq:    torch.Tensor,  # (B, T, D)
        mask:   torch.Tensor,  # (B, T) bool  True=valid
    ) -> torch.Tensor:         # (B, D)
        if seq.shape[1] == 0:
            return torch.zeros_like(target)
        B, T, D = seq.shape
        # Expand target to (B, T, D) for concat
        t_exp = target.unsqueeze(1).expand(B, T, D)
        inp   = torch.cat([t_exp, seq], dim=-1)   # (B, T, 2D)
        scores = self.score(inp).squeeze(-1)       # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights)
        return (weights.unsqueeze(-1) * seq).sum(dim=1)   # (B, D)


class MultiViewReadout(nn.Module):
    """
    Produces four representation vectors from the transformer output:
      - target_repr : x[:, 0]  (target item token)
      - user_repr   : masked mean-pool over TT_USER tokens
      - item_repr   : masked mean-pool over TT_ITEM tokens
      - din_repr    : DIN attention(target, seq tokens)

    Final input to the MLP head:
      concat(target, user, item, din)  +  user ⊙ item  →  5 × D
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.din = DINAttention(cfg.d_model)

    @staticmethod
    def _masked_mean(x, mask):
        """x: (B,T,D)  mask: (B,T) bool → (B,D)"""
        mask_f = mask.float().unsqueeze(-1)          # (B, T, 1)
        denom  = mask_f.sum(dim=1).clamp(min=1e-6)   # (B, 1)
        return (x * mask_f).sum(dim=1) / denom       # (B, D)

    def forward(self, x, types, mask):
        """
        x:     (B, T, D)
        types: (B, T)  long
        mask:  (B, T)  bool  True=valid
        """
        target_repr = x[:, 0]   # always slot 0

        # Masks for each token group
        user_mask = mask & (types == TT_USER)
        item_mask = mask & (types == TT_ITEM)
        seq_mask  = mask & (
            (types == TT_ACT_SEQ) | (types == TT_CONT_SEQ) | (types == TT_ITEM_SEQ)
        )

        user_repr = self._masked_mean(x, user_mask)
        item_repr = self._masked_mean(x, item_mask)

        # DIN over all seq tokens
        din_repr = self.din(target_repr, x, seq_mask)

        # Explicit cross: user ⊙ item
        cross = user_repr * item_repr

        # Final concat: 5D
        combined = torch.cat([target_repr, user_repr, item_repr, din_repr, cross], dim=-1)
        return combined


class EnhancedRecommender(nn.Module):
    """
    Full model:
      TokenEmbedding
        → 4 × HeterogeneousTransformerLayer  (stochastic depth)
          → LayerNorm
            → MultiViewReadout (target + user + item + DIN + cross)
              → Deep MLP head → CVR logit
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.emb    = TokenEmbedding(cfg)

        # Linearly increasing stochastic depth rates across layers
        dp_rates = [cfg.drop_path * i / max(cfg.n_layers - 1, 1)
                    for i in range(cfg.n_layers)]
        self.layers = nn.ModuleList([
            HetTransformerLayer(cfg, dp_rates[i]) for i in range(cfg.n_layers)
        ])
        self.norm    = nn.LayerNorm(cfg.d_model)
        self.readout = MultiViewReadout(cfg)

        # Head input dim: 5 × d_model  (target + user + item + din + cross)
        head_in = cfg.d_model * 5
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.d_model * 2), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.d_model), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model // 2), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, ids, vals, types, pos, mask):
        x = self.emb(ids, vals, types, pos)
        for layer in self.layers:
            x = layer(x, types, mask)
        x        = self.norm(x)
        combined = self.readout(x, types, mask)
        return self.head(combined).squeeze(-1)


# ════════════════════════════════════════════════════════════
# 4. Loss Functions
# ════════════════════════════════════════════════════════════

def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    label_smooth: float = 0.05,
) -> torch.Tensor:
    """
    Focal Binary Cross-Entropy with label smoothing.

    - Label smoothing prevents over-confidence by mixing targets with ε/2.
    - Focal weighting (1 - p_t)^γ down-weights easy examples,
      focusing learning on hard/rare positives.
    """
    # Label smoothing
    smooth_targets = targets * (1.0 - label_smooth) + 0.5 * label_smooth

    # Standard BCE per element
    bce = F.binary_cross_entropy_with_logits(
        logits, smooth_targets, reduction="none"
    )

    # Focal weight
    if gamma > 0:
        probs  = torch.sigmoid(logits)
        p_t    = probs * targets + (1 - probs) * (1 - targets)
        weight = (1.0 - p_t) ** gamma
        bce    = weight * bce

    return bce.mean()


# ════════════════════════════════════════════════════════════
# 5. Training & Evaluation
# ════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, grad_clip, cfg):
    model.train()
    total_loss = 0.0
    preds, labels = [], []

    for batch in loader:
        ids   = batch["ids"].to(device)
        vals  = batch["vals"].to(device)
        types = batch["types"].to(device)
        pos   = batch["pos"].to(device)
        mask  = batch["mask"].to(device)
        y     = batch["labels"].to(device)

        logits = model(ids, vals, types, pos, mask)
        loss   = focal_bce_loss(
            logits, y, gamma=cfg.focal_gamma, label_smooth=cfg.label_smooth
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        preds.extend(torch.sigmoid(logits).detach().cpu().tolist())
        labels.extend(y.cpu().tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return total_loss / len(loader), auc


@torch.no_grad()
def eval_epoch(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []

    for batch in loader:
        ids   = batch["ids"].to(device)
        vals  = batch["vals"].to(device)
        types = batch["types"].to(device)
        pos   = batch["pos"].to(device)
        mask  = batch["mask"].to(device)
        y     = batch["labels"].to(device)

        logits = model(ids, vals, types, pos, mask)
        loss   = focal_bce_loss(
            logits, y, gamma=cfg.focal_gamma, label_smooth=cfg.label_smooth
        )

        total_loss += loss.item()
        preds.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(y.cpu().tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return total_loss / len(loader), auc


# ════════════════════════════════════════════════════════════
# 6. Main
# ════════════════════════════════════════════════════════════

def auto_cvr_type(df: pd.DataFrame) -> int:
    for t in [1, 2, 3]:
        pos_rate = df["label"].apply(lambda x: parse_label(x, t)).mean()
        print(f"  action_type >= {t}: pos_rate = {pos_rate:.3f}")
        if 0.01 <= pos_rate <= 0.99:
            print(f"  → Using cvr_type = {t}  (pos_rate={pos_rate:.3f})")
            return t
    print("  Warning: could not find non-degenerate label, using cvr_type=1")
    return 1


def print_data_diagnostics(df: pd.DataFrame):
    print("\n── Dataset Diagnostics ──────────────────────────────")
    print(f"  Rows: {len(df)}")
    row = df.iloc[0]
    print(f"  user_feature length : {len(row['user_feature'])}")
    print(f"  item_feature length : {len(row['item_feature'])}")
    seq = row["seq_feature"]
    for key in ["action_seq", "content_seq", "item_seq"]:
        sub = _get(seq, key, None)
        n   = len(sub) if sub is not None else 0
        if sub is not None and n > 0:
            arr = _get(sub[0], "int_array", None)
            seq_len = len(arr) if arr is not None else "?"
        else:
            seq_len = 0
        print(f"  seq_feature.{key}: {n} feature(s), seq_len≈{seq_len}")
    lbl = row["label"]
    print(f"  label sample        : {list(lbl)[:3]}")
    print("─────────────────────────────────────────────────────\n")


def main():
    cfg = Config(
        d_model=128, n_layers=4, n_heads=8, d_ff=512,
        dropout=0.15, drop_path=0.10,
        vocab_size=32768, max_seq_len=50, max_field_t=48,
        batch_size=256, lr=3e-4, wd=1e-4,
        epochs=60, grad_clip=1.0, warmup_epochs=4,
        focal_gamma=2.0, label_smooth=0.05,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")

    print("Loading sample_data.parquet ...")
    df = pd.read_parquet("sample_data.parquet")
    print(f"Shape  : {df.shape}")
    print_data_diagnostics(df)

    print("Label distribution:")
    cfg.cvr_type = auto_cvr_type(df)

    n     = len(df)
    split = int(0.8 * n)
    df_tr = df.iloc[:split].reset_index(drop=True)
    df_va = df.iloc[split:].reset_index(drop=True)
    print(f"\nTrain: {len(df_tr)} samples | Val: {len(df_va)} samples")

    pos_tr = df_tr["label"].apply(lambda x: parse_label(x, cfg.cvr_type)).mean()
    pos_va = df_va["label"].apply(lambda x: parse_label(x, cfg.cvr_type)).mean()
    print(f"Train pos_rate: {pos_tr:.3f} | Val pos_rate: {pos_va:.3f}")

    print("\nBuilding datasets:")
    tr_ds = TAADataset(df_tr, cfg)
    print("\nBuilding datasets:")
    va_ds = TAADataset(df_va, cfg)

    _col  = partial(collate_fn, cfg=cfg)
    tr_ld = DataLoader(tr_ds, cfg.batch_size, shuffle=True,
                       collate_fn=_col, num_workers=0, pin_memory=(device.type == "cuda"))
    va_ld = DataLoader(va_ds, cfg.batch_size, shuffle=False,
                       collate_fn=_col, num_workers=0, pin_memory=(device.type == "cuda"))

    model    = EnhancedRecommender(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters : {n_params:,}")
    print(f"\nArchitecture:")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, d_ff={cfg.d_ff}")
    print(f"  vocab_size={cfg.vocab_size}, drop_path={cfg.drop_path}")
    print(f"  focal_gamma={cfg.focal_gamma}, label_smooth={cfg.label_smooth}")
    print(f"  Readout: target + user_pool + item_pool + DIN + user⊙item")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.wd
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.05, total_iters=cfg.warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.lr * 0.01
            ),
        ],
        milestones=[cfg.warmup_epochs],
    )

    best_auc   = 0.0
    best_epoch = 0

    print(f"\n{'Epoch':>6}  {'Tr Loss':>8}  {'Tr AUC':>8}  "
          f"{'Va Loss':>8}  {'Va AUC':>8}  {'LR':>10}")
    print("─" * 64)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_auc = train_epoch(model, tr_ld, optimizer, device, cfg.grad_clip, cfg)
        va_loss, va_auc = eval_epoch(model, va_ld, device, cfg)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        flag   = " ✓" if va_auc > best_auc else ""
        print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_auc:>8.4f}  "
              f"{va_loss:>8.4f}  {va_auc:>8.4f}  {lr_now:>10.2e}{flag}")

        if va_auc > best_auc:
            best_auc   = va_auc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "config": cfg,
                    "state_dict": model.state_dict(),
                    "val_auc": best_auc,
                },
                "best_model_v2.pt",
            )

    print("─" * 64)
    print(f"\nBest Val AUC: {best_auc:.4f}  (epoch {best_epoch})")
    print("Model saved to best_model_v2.pt")


if __name__ == "__main__":
    main()