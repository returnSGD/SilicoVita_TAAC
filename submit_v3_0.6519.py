#!/usr/bin/env python3
"""
UniSFIN v3: Unified Sequential Feature Interaction Network
TAAC2026 Competition — Single-Model Advanced Architecture

═══════════════════════════════════════════════════════════════════════
 Architecture Overview
═══════════════════════════════════════════════════════════════════════

  TokenEmbedding (FeatIDEmb ⊙ ValueGate + TypeEmb + PosEmb)
    ↓
  N × UniFormerBlock  ← [UNIFIED BLOCK INNOVATION]
    │  ┌─ Pre-LN Multi-Head Self-Attention + HetBias
    │  │    (all token types attend across paradigms)
    │  ├─ Pre-LN SwiGLU FFN
    │  └─ StochasticDepth per branch
    ↓
  LayerNorm
    ↓
  MultiViewBIReadout  ← [BEHAVIORAL INTERACTION READOUT]
    │  ┌─ target_repr   = x[:,0]  (target item)
    │  ├─ user_pool     = AttentionPooling(x[user_tokens], target)
    │  ├─ item_pool     = AttentionPooling(x[item_tokens], target)
    │  ├─ din_act       = PositionWeightedDIN(target, x[action_seq])
    │  ├─ din_cnt       = PositionWeightedDIN(target, x[content_seq])
    │  ├─ din_itm       = PositionWeightedDIN(target, x[item_seq])
    │  ├─ fm_repr       = FM2ndOrder(x[user_tokens ∪ item_tokens])
    │  └─ cross         = user_pool ⊙ item_pool
    ↓
  Deep MLP Head (BN+GELU) → CVR logit

═══════════════════════════════════════════════════════════════════════
 Key Innovations over v2 (AUC ~0.6856)
═══════════════════════════════════════════════════════════════════════
  1. SwiGLU FFN (Swish-Gated Linear Unit) - richer gating vs GELU
  2. Three dedicated PositionWeightedDIN modules (one per seq type)
       each computes attention(target, seq_tok) + log(position_decay)
       so recency and relevance are jointly scored
  3. AttentionPooling for user/item field representations
       (target-aware query instead of plain mean pooling)
  4. FM Second-Order Readout on post-attention field embeddings
       captures residual pair-wise interactions not modeled by attention
  5. Float value extraction from sequence features
       (use actual float_array values as embedding gates, not constant 1.0)
  6. Larger model: d_model=192, n_layers=6, vocab=65536
  7. OneCycleLR scheduler with 6% warmup
  8. SiLU in value gate (smoother than ReLU)
  9. Batch Normalization in prediction head (more stable deep MLP)
 10. Layer-wise stochastic depth (linearly increasing drop rate)
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

TT_TARGET = 0
TT_USER = 1
TT_ITEM = 2
TT_ACT_SEQ = 3
TT_CONT_SEQ = 4
TT_ITEM_SEQ = 5
N_TT = 6


@dataclass
class Config:
    # ── Embedding ──────────────────────────────────────────
    d_model: int = 192  # ↑ from 128 (v2)
    vocab_size: int = 65536  # ↑ from 32768 (fewer hash collisions)
    max_seq_len: int = 80  # ↑ from 50
    max_field_t: int = 64  # ↑ from 48

    # ── Transformer ────────────────────────────────────────
    n_layers: int = 6  # ↑ from 4
    n_heads: int = 8
    d_ff: int = 512  # SwiGLU inner dim
    dropout: float = 0.12
    drop_path: float = 0.15  # max stochastic depth rate

    # ── Training ───────────────────────────────────────────
    batch_size: int = 64
    lr: float = 1e-4
    wd: float = 1e-4
    epochs: int = 60
    grad_clip: float = 1.0
    pct_start: float = 0.06  # OneCycleLR warmup fraction

    # ── Loss ───────────────────────────────────────────────
    focal_gamma: float = 2.0
    label_smooth: float = 0.03

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
        feat_arr, max_tokens: int, vocab_size: int = 65536
) -> Tuple[List[int], List[float]]:
    fids: List[int] = []
    fvals: List[float] = []
    if feat_arr is None:
        return fids, fvals

    for feat in feat_arr:
        fid = int(_get(feat, "feature_id", 0)) % (vocab_size - 1) + 1
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
        seq_feature, sub_key: str, max_seq_len: int, vocab_size: int = 65536
) -> Tuple[List[int], List[int], List[float]]:
    """
    Parse sequence sub-type. Returns (tok_ids, positions, float_vals).

    Enhancement over v2:
    - Also extracts float_array values as embedding gate scalars.
      If float_array[t] exists, use normalize_val(float_array[t]) as the
      gate value; otherwise default to 1.0 (neutral gate).
    - Uses all feature structs (not just first) up to max_seq_len.
    """
    if seq_feature is None:
        return [], [], []
    sub = _get(seq_feature, sub_key, None)
    if sub is None or len(sub) == 0:
        return [], [], []

    tok_ids: List[int] = []
    positions: List[int] = []
    float_vals: List[float] = []

    for feat in sub:
        int_arr = _get(feat, "int_array", None)
        flt_arr = _get(feat, "float_array", None)

        if int_arr is not None:
            for pos, v in enumerate(int_arr[:max_seq_len]):
                tok_ids.append(int(v) % (vocab_size - 1) + 1)
                positions.append(pos + 1)  # 1-indexed; 1 = first/most-recent slot
                if flt_arr is not None and pos < len(flt_arr):
                    float_vals.append(normalize_val(float(flt_arr[pos])))
                else:
                    float_vals.append(1.0)

        if len(tok_ids) >= max_seq_len:
            break

    return tok_ids[:max_seq_len], positions[:max_seq_len], float_vals[:max_seq_len]


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
        avg_t = np.mean([self._count(s) for s in self.samples])
        print(f"  Done. Avg tokens/sample: {avg_t:.1f}")

    @staticmethod
    def _count(s: Dict) -> int:
        return (1 + len(s["u_fids"]) + len(s["i_fids"])
                + len(s["act_ids"]) + len(s["cnt_ids"]) + len(s["itm_ids"]))

    def _process(self, row) -> Dict:
        cfg = self.cfg
        label = parse_label(row["label"], cfg.cvr_type)
        target_id = int(row["item_id"]) % (cfg.vocab_size - 1) + 1

        u_fids, u_fvals = parse_feature_array(
            row["user_feature"], cfg.max_field_t, cfg.vocab_size)
        i_fids, i_fvals = parse_feature_array(
            row["item_feature"], cfg.max_field_t, cfg.vocab_size)

        seq = row["seq_feature"]
        act_ids, act_pos, act_vals = parse_seq_sub(
            seq, "action_seq", cfg.max_seq_len, cfg.vocab_size)
        cnt_ids, cnt_pos, cnt_vals = parse_seq_sub(
            seq, "content_seq", cfg.max_seq_len, cfg.vocab_size)
        itm_ids, itm_pos, itm_vals = parse_seq_sub(
            seq, "item_seq", cfg.max_seq_len, cfg.vocab_size)

        return dict(
            label=label, target_id=target_id,
            u_fids=u_fids, u_fvals=u_fvals,
            i_fids=i_fids, i_fvals=i_fvals,
            act_ids=act_ids, act_pos=act_pos, act_vals=act_vals,
            cnt_ids=cnt_ids, cnt_pos=cnt_pos, cnt_vals=cnt_vals,
            itm_ids=itm_ids, itm_pos=itm_pos, itm_vals=itm_vals,
        )

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]


def collate_fn(batch: List[Dict], cfg: Config) -> Dict:
    """
    Assemble batch into padded tensors.

    Token layout per sample:
      [target] | [user_fields] | [item_fields]
               | [action_seq]  | [content_seq] | [item_seq]

    Outputs (B × T_max):
      ids   : feature/item IDs (long)
      vals  : normalized scalar values (float)  ← now uses actual seq float vals
      types : token type [0-5] (long)
      pos   : position within seq, 0 = non-sequential (long)
      mask  : True = valid token (bool)
    """
    B = len(batch)
    all_ids, all_vals, all_types, all_pos = [], [], [], []

    for s in batch:
        ids, vals, types, pos = [], [], [], []

        def add(i, v, t, p):
            ids.append(i);
            vals.append(v);
            types.append(t);
            pos.append(p)

        add(s["target_id"], 1.0, TT_TARGET, 0)

        for fid, fv in zip(s["u_fids"], s["u_fvals"]):
            add(fid, fv, TT_USER, 0)
        for fid, fv in zip(s["i_fids"], s["i_fvals"]):
            add(fid, fv, TT_ITEM, 0)

        for tid, p, fv in zip(s["act_ids"], s["act_pos"], s["act_vals"]):
            add(tid, fv, TT_ACT_SEQ, p)
        for tid, p, fv in zip(s["cnt_ids"], s["cnt_pos"], s["cnt_vals"]):
            add(tid, fv, TT_CONT_SEQ, p)
        for tid, p, fv in zip(s["itm_ids"], s["itm_pos"], s["itm_vals"]):
            add(tid, fv, TT_ITEM_SEQ, p)

        all_ids.append(ids);
        all_vals.append(vals)
        all_types.append(types);
        all_pos.append(pos)

    T = max(len(x) for x in all_ids)
    t_ids = torch.zeros(B, T, dtype=torch.long)
    t_vals = torch.zeros(B, T, dtype=torch.float)
    t_types = torch.zeros(B, T, dtype=torch.long)
    t_pos = torch.zeros(B, T, dtype=torch.long)
    t_mask = torch.zeros(B, T, dtype=torch.bool)

    for i in range(B):
        L = len(all_ids[i])
        t_ids[i, :L] = torch.tensor(all_ids[i], dtype=torch.long)
        t_vals[i, :L] = torch.tensor(all_vals[i], dtype=torch.float)
        t_types[i, :L] = torch.tensor(all_types[i], dtype=torch.long)
        t_pos[i, :L] = torch.tensor(all_pos[i], dtype=torch.long)
        t_mask[i, :L] = True

    labels = torch.tensor([s["label"] for s in batch], dtype=torch.float)
    return dict(ids=t_ids, vals=t_vals, types=t_types,
                pos=t_pos, mask=t_mask, labels=labels)


# ════════════════════════════════════════════════════════════
# 3. Model Components
# ════════════════════════════════════════════════════════════

class TokenEmbedding(nn.Module):
    """
    Unified Feature-as-Token embedding:

      emb(x) = FeatIDEmb(id) ⊙ ValueGate(v)    ← content representation
             + TokenTypeEmb(type)               ← paradigm identity
             + PositionEmb(pos)                 ← temporal order (seq only)

    ValueGate uses SiLU (smoother gradient than ReLU) for the hidden layer.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        self.feat_emb = nn.Embedding(cfg.vocab_size, D, padding_idx=0)
        self.type_emb = nn.Embedding(N_TT, D)
        self.pos_emb = nn.Embedding(cfg.max_seq_len + 2, D)

        self.val_gate = nn.Sequential(
            nn.Linear(1, D // 4), nn.SiLU(),  # SiLU replaces ReLU
            nn.Linear(D // 4, D), nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(D)
        self.dropout = nn.Dropout(cfg.dropout)
        nn.init.normal_(self.feat_emb.weight, std=0.02)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, ids, vals, types, pos):
        pos = pos.clamp(0, self.pos_emb.num_embeddings - 1)
        gate = self.val_gate(vals.unsqueeze(-1))  # (B,T,D) in (0,1)
        x = self.feat_emb(ids) * gate  # value-gated feature
        x = x + self.type_emb(types)  # + paradigm identity
        x = x + self.pos_emb(pos)  # + temporal position
        return self.dropout(self.norm(x))


class HetBias(nn.Module):
    """
    Heterogeneous attention bias B[h, type_q, type_k].
    Added to attention logits before softmax, steering cross-paradigm
    attention patterns (e.g., how much target attends to seq tokens).
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.H = n_heads
        self.bias = nn.Parameter(torch.zeros(n_heads, N_TT, N_TT))

    def forward(self, tq: torch.Tensor, tk: torch.Tensor) -> torch.Tensor:
        B, Tq = tq.shape;
        Tk = tk.shape[1];
        H = self.H
        tq_bh = tq.unsqueeze(1).expand(B, H, Tq).reshape(B * H, Tq)
        tk_bh = tk.unsqueeze(1).expand(B, H, Tk).reshape(B * H, Tk)
        bias = (self.bias.unsqueeze(0).expand(B, H, N_TT, N_TT)
                .reshape(B * H, N_TT, N_TT))
        row_idx = tq_bh.unsqueeze(-1).expand(B * H, Tq, N_TT)
        rows = bias.gather(1, row_idx)
        col_idx = tk_bh.unsqueeze(1).expand(B * H, Tq, Tk)
        return rows.gather(2, col_idx).reshape(B, H, Tq, Tk)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (used in LLaMA, PaLM, etc.).

    output = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )

    SwiGLU provides a multiplicative gating mechanism that
    outperforms vanilla GELU FFN with similar parameter count.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


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
        noise = torch.empty(shape, dtype=x.dtype, device=x.device)
        noise.bernoulli_(keep).div_(keep)
        return x * noise


class UniFormerBlock(nn.Module):
    """
    Unified Former Block — the homogeneous stackable backbone unit.

    Bridges sequential and feature-interaction paradigms in a single block:

      x → LN → MHA(Q,K,V) + HetBias(type_q, type_k) → StochasticDepth → + x
        → LN → SwiGLU FFN                             → StochasticDepth → + x

    HetBias steers inter-paradigm attention (how much feature tokens attend
    to sequence tokens and vice versa), making this a genuinely unified block
    rather than separate feature-interaction and sequence-modeling branches.
    """

    def __init__(self, cfg: Config, drop_path_prob: float = 0.0):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, "d_model must be divisible by n_heads"
        self.H = H
        self.Dh = D // H

        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out = nn.Linear(D, D)
        self.het = HetBias(H)
        self.ffn = SwiGLUFFN(D, cfg.d_ff, cfg.dropout)

        self.n1 = nn.LayerNorm(D)
        self.n2 = nn.LayerNorm(D)
        self.dp = nn.Dropout(cfg.dropout)
        self.sd = StochasticDepth(drop_path_prob)

    def forward(
            self,
            x: torch.Tensor,  # (B, T, D)
            types: torch.Tensor,  # (B, T)
            mask: Optional[torch.Tensor],  # (B, T) bool — True = valid
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.H, self.Dh

        # ── Multi-Head Self-Attention (Pre-Norm) ─────────────
        residual = x
        xn = self.n1(x)
        Q, K, V = self.qkv(xn).split(D, dim=-1)

        def sh(t): return t.view(B, T, H, Dh).transpose(1, 2)

        Q, K, V = sh(Q), sh(K), sh(V)

        logits = (Q @ K.transpose(-2, -1)) / math.sqrt(Dh)
        logits += self.het(types, types)

        if mask is not None:
            logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))

        w = torch.nan_to_num(F.softmax(logits, dim=-1))
        w = self.dp(w)
        attn = (w @ V).transpose(1, 2).reshape(B, T, D)
        x = residual + self.sd(self.dp(self.out(attn)))

        # ── SwiGLU Feed-Forward (Pre-Norm) ───────────────────
        x = x + self.sd(self.ffn(self.n2(x)))
        return x


class AttentionPooling(nn.Module):
    """
    Target-query attention pooling for field token aggregation.

    Instead of mean-pooling, uses the target item representation as a
    query to compute token-level importance scores, producing a
    context-aware aggregation of the field tokens.

    score_t = (W_q * target) · (W_k * token_t) / sqrt(D)
    output  = Σ_t softmax(score_t) * token_t
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5

    def forward(
            self,
            query: torch.Tensor,  # (B, D) — target repr
            tokens: torch.Tensor,  # (B, T, D)
            mask: torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:  # (B, D)
        if tokens.shape[1] == 0 or mask.sum() == 0:
            return torch.zeros_like(query)
        q = self.Wq(query).unsqueeze(1)  # (B, 1, D)
        k = self.Wk(tokens)  # (B, T, D)
        scores = (q * k).sum(-1) * self.scale  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.nan_to_num(F.softmax(scores, dim=-1))
        return (w.unsqueeze(-1) * tokens).sum(1)  # (B, D)


class PositionWeightedDIN(nn.Module):
    """
    Deep Interest Network with learnable position-decay weighting.

    Each sequence token is scored jointly by:
      (a) Relevance:  MLP( concat[target, seq_tok, target - seq_tok] ) → scalar
      (b) Recency:    log( exp(-λ * (pos - 1)) )  where λ is learned

    The learnable decay λ discovers whether earlier or later positions
    in each sequence sub-type are more informative for CVR prediction.

    Having three separate DIN modules (action / content / item) allows
    each behavioral stream to learn its own relevance and recency patterns.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
        )
        # log_decay: log(λ). Init at -2 → λ ≈ 0.135 (gentle decay by default)
        self.log_decay = nn.Parameter(torch.tensor(-2.0))

    def forward(
            self,
            target: torch.Tensor,  # (B, D)
            seq: torch.Tensor,  # (B, T, D)
            positions: torch.Tensor,  # (B, T) int  1-indexed
            mask: torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:  # (B, D)
        if seq.shape[1] == 0 or mask.sum() == 0:
            return torch.zeros_like(target)

        B, T, D = seq.shape
        t_exp = target.unsqueeze(1).expand(B, T, D)
        diff = t_exp - seq
        inp = torch.cat([t_exp, seq, diff], dim=-1)  # (B, T, 3D)
        scores = self.score(inp).squeeze(-1)  # (B, T)

        # Recency weighting: λ = exp(log_decay), positions-1 gives 0-indexed age
        decay = torch.exp(self.log_decay)
        age = (positions.float() - 1).clamp(min=0)
        pos_log_w = -decay * age  # (B, T) log weights
        scores = scores + pos_log_w

        scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.nan_to_num(F.softmax(scores, dim=-1))
        return (w.unsqueeze(-1) * seq).sum(dim=1)  # (B, D)


class MultiViewBIReadout(nn.Module):
    """
    Multi-View Behavioral Interaction Readout.

    Produces 8 complementary representation vectors from transformer output,
    then concatenates them for the prediction head.

    View 1 — target_repr:  x[:,0]  (target item's final representation)
    View 2 — user_pool:    AttentionPooling(target, user_field_tokens)
    View 3 — item_pool:    AttentionPooling(target, item_field_tokens)
    View 4 — din_act:      PositionWeightedDIN(target, action_seq_tokens)
    View 5 — din_cnt:      PositionWeightedDIN(target, content_seq_tokens)
    View 6 — din_itm:      PositionWeightedDIN(target, item_seq_tokens)
    View 7 — fm_repr:      FM2ndOrder(user_tokens ∪ item_tokens)
                           = 0.5[(Σe_i)² - Σe_i²]  on post-attention embeddings
                           captures residual pair-wise feature interactions
    View 8 — cross:        user_pool ⊙ item_pool  (user-item affinity)

    Total: 8 × D fed into the prediction MLP.

    fm_repr uses the transformer-refined embeddings (not raw embeddings),
    making it an "Attention-Transformed FM" that models high-order residual
    interactions remaining after the self-attention layers.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        self.user_pool = AttentionPooling(D)
        self.item_pool = AttentionPooling(D)
        self.din_act = PositionWeightedDIN(D)
        self.din_cnt = PositionWeightedDIN(D)
        self.din_itm = PositionWeightedDIN(D)

    @staticmethod
    def _fm2(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """FM second-order pooling: (B,T,D) + (B,T) → (B,D)"""
        mf = mask.float().unsqueeze(-1)  # (B,T,1)
        ex = x * mf  # (B,T,D) zeroed outside mask
        s_sq = ex.sum(1) ** 2  # (B,D)
        sq_s = (ex ** 2).sum(1)  # (B,D)
        return 0.5 * (s_sq - sq_s)  # (B,D)

    def forward(
            self,
            x: torch.Tensor,  # (B, T, D)
            types: torch.Tensor,  # (B, T)
            mask: torch.Tensor,  # (B, T) bool
            pos: torch.Tensor,  # (B, T) long  — raw position indices
    ) -> torch.Tensor:  # (B, 8*D)

        target_repr = x[:, 0]  # slot 0 is always the target token

        # Per-type masks (valid token AND correct type)
        u_mask = mask & (types == TT_USER)
        i_mask = mask & (types == TT_ITEM)
        act_mask = mask & (types == TT_ACT_SEQ)
        cnt_mask = mask & (types == TT_CONT_SEQ)
        itm_mask = mask & (types == TT_ITEM_SEQ)
        field_mask = mask & ((types == TT_USER) | (types == TT_ITEM))

        # Views 2 & 3: target-aware attention pooling over field tokens
        user_pool = self.user_pool(target_repr, x, u_mask)
        item_pool = self.item_pool(target_repr, x, i_mask)

        # Views 4-6: three independent position-weighted DIN modules
        din_act = self.din_act(target_repr, x, pos, act_mask)
        din_cnt = self.din_cnt(target_repr, x, pos, cnt_mask)
        din_itm = self.din_itm(target_repr, x, pos, itm_mask)

        # View 7: FM second-order on post-attention field embeddings
        fm_repr = self._fm2(x, field_mask)

        # View 8: Hadamard user-item cross product (affinity signal)
        cross = user_pool * item_pool

        return torch.cat(
            [target_repr, user_pool, item_pool,
             din_act, din_cnt, din_itm, fm_repr, cross], dim=-1
        )  # (B, 8*D)


class UniSFIN(nn.Module):
    """
    Unified Sequential Feature Interaction Network.

    Full model:
      TokenEmbedding
        → N × UniFormerBlock  (linearly increasing stochastic depth)
          → LayerNorm
            → MultiViewBIReadout  (8 complementary views)
              → Deep MLP with BatchNorm → CVR logit
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.emb = TokenEmbedding(cfg)

        # Linearly increasing stochastic depth rates (0 → max_drop_path)
        dp_rates = [cfg.drop_path * i / max(cfg.n_layers - 1, 1)
                    for i in range(cfg.n_layers)]
        self.blocks = nn.ModuleList([
            UniFormerBlock(cfg, dp_rates[i]) for i in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.readout = MultiViewBIReadout(cfg)

        # Head: 8*D → 512 → 256 → 128 → 1
        # BatchNorm1d stabilizes deep MLP training for tabular-style inputs
        D = cfg.d_model
        self.head = nn.Sequential(
            nn.Linear(8 * D, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout * 0.5),
            nn.Linear(128, 1),
        )

    def forward(
            self,
            ids: torch.Tensor,  # (B, T)
            vals: torch.Tensor,  # (B, T)
            types: torch.Tensor,  # (B, T)
            pos: torch.Tensor,  # (B, T)
            mask: torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:  # (B,) logits

        x = self.emb(ids, vals, types, pos)
        for block in self.blocks:
            x = block(x, types, mask)
        x = self.norm(x)
        combined = self.readout(x, types, mask, pos)
        return self.head(combined).squeeze(-1)


# ════════════════════════════════════════════════════════════
# 4. Loss Function
# ════════════════════════════════════════════════════════════

def focal_bce_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        label_smooth: float = 0.03,
) -> torch.Tensor:
    """
    Focal Binary Cross-Entropy with Label Smoothing.

    - Label smoothing: mixes hard 0/1 targets with ε/2,
      preventing the model from being over-confident.
    - Focal weight (1 - p_t)^γ: down-weights easy/majority examples,
      focusing gradient on hard positives (CVR conversion events).
    """
    smooth_targets = targets * (1.0 - label_smooth) + 0.5 * label_smooth
    bce = F.binary_cross_entropy_with_logits(
        logits, smooth_targets, reduction="none")

    if gamma > 0:
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        weight = (1.0 - p_t) ** gamma
        bce = weight * bce

    return bce.mean()


# ════════════════════════════════════════════════════════════
# 5. Training & Evaluation
# ════════════════════════════════════════════════════════════

def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cfg: Config,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    preds, labels = [], []

    for batch in loader:
        ids = batch["ids"].to(device)
        vals = batch["vals"].to(device)
        types = batch["types"].to(device)
        pos = batch["pos"].to(device)
        mask = batch["mask"].to(device)
        y = batch["labels"].to(device)

        logits = model(ids, vals, types, pos, mask)
        loss = focal_bce_loss(logits, y, cfg.focal_gamma, cfg.label_smooth)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        preds.extend(torch.sigmoid(logits).detach().cpu().tolist())
        labels.extend(y.cpu().tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return total_loss / len(loader), auc


@torch.no_grad()
def eval_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        cfg: Config,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    preds, labels = [], []

    for batch in loader:
        ids = batch["ids"].to(device)
        vals = batch["vals"].to(device)
        types = batch["types"].to(device)
        pos = batch["pos"].to(device)
        mask = batch["mask"].to(device)
        y = batch["labels"].to(device)

        logits = model(ids, vals, types, pos, mask)
        loss = focal_bce_loss(logits, y, cfg.focal_gamma, cfg.label_smooth)

        total_loss += loss.item()
        preds.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(y.cpu().tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return total_loss / len(loader), auc


# ════════════════════════════════════════════════════════════
# 6. Helpers & Main
# ════════════════════════════════════════════════════════════

def auto_cvr_type(df: pd.DataFrame) -> int:
    for t in [1, 2, 3]:
        pos_rate = df["label"].apply(lambda x: parse_label(x, t)).mean()
        print(f"  action_type >= {t}: pos_rate = {pos_rate:.3f}")
        if 0.01 <= pos_rate <= 0.99:
            print(f"  → Using cvr_type = {t}  (pos_rate={pos_rate:.3f})")
            return t
    print("  Warning: falling back to cvr_type=1")
    return 1


def print_diagnostics(df: pd.DataFrame):
    print("\n── Dataset Diagnostics ──────────────────────────────────")
    print(f"  Rows: {len(df)}")
    row = df.iloc[0]
    print(f"  user_feature length : {len(row['user_feature'])}")
    print(f"  item_feature length : {len(row['item_feature'])}")
    seq = row["seq_feature"]
    for key in ["action_seq", "content_seq", "item_seq"]:
        sub = _get(seq, key, None)
        n = len(sub) if sub is not None else 0
        if sub is not None and n > 0:
            arr = _get(sub[0], "int_array", None)
            seq_len = len(arr) if arr is not None else "?"
            has_flt = _get(sub[0], "float_array", None) is not None
        else:
            seq_len = 0;
            has_flt = False
        print(f"  seq_feature.{key:<12}: {n} struct(s), "
              f"int_array len≈{seq_len}, float_array={has_flt}")
    print(f"  label sample        : {list(row['label'])[:3]}")
    print("─────────────────────────────────────────────────────────\n")


def main():
    cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")

    # ── Load ──────────────────────────────────────────────────────────
    print("Loading sample_data.parquet ...")
    df = pd.read_parquet("sample_data.parquet")
    print(f"Shape  : {df.shape}")
    print_diagnostics(df)

    print("Label distribution:")
    cfg.cvr_type = auto_cvr_type(df)

    # ── Train / Val split (80/20, time-ordered) ───────────────────────
    n = len(df)
    split = int(0.8 * n)
    df_tr = df.iloc[:split].reset_index(drop=True)
    df_va = df.iloc[split:].reset_index(drop=True)
    print(f"\nTrain: {len(df_tr)} samples | Val: {len(df_va)} samples")

    pos_tr = df_tr["label"].apply(lambda x: parse_label(x, cfg.cvr_type)).mean()
    pos_va = df_va["label"].apply(lambda x: parse_label(x, cfg.cvr_type)).mean()
    print(f"Train pos_rate: {pos_tr:.3f} | Val pos_rate: {pos_va:.3f}")

    # ── Datasets & DataLoaders ────────────────────────────────────────
    print("\nBuilding train dataset:")
    tr_ds = TAADataset(df_tr, cfg)
    print("\nBuilding val dataset:")
    va_ds = TAADataset(df_va, cfg)

    _col = partial(collate_fn, cfg=cfg)
    tr_ld = DataLoader(tr_ds, cfg.batch_size, shuffle=True,
                       collate_fn=_col, num_workers=2,
                       pin_memory=(device.type == "cuda"))
    va_ld = DataLoader(va_ds, cfg.batch_size, shuffle=False,
                       collate_fn=_col, num_workers=2,
                       pin_memory=(device.type == "cuda"))

    # ── Model ─────────────────────────────────────────────────────────
    model = UniSFIN(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel       : UniSFIN v3")
    print(f"Parameters  : {n_params:,}")
    print(f"d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, d_ff={cfg.d_ff} (SwiGLU)")
    print(f"vocab={cfg.vocab_size}, max_seq={cfg.max_seq_len}, "
          f"drop_path_max={cfg.drop_path}")
    print(f"Readout: 8 × {cfg.d_model}D "
          f"[target|user_pool|item_pool|din×3|fm|cross]")

    # ── Optimizer + OneCycleLR ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.wd,
        betas=(0.9, 0.999), eps=1e-8,
    )
    total_steps = cfg.epochs * len(tr_ld)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=cfg.pct_start,  # 6% warmup
        anneal_strategy="cos",
        div_factor=25.0,  # initial_lr = max_lr / 25
        final_div_factor=1e4,  # final_lr  = max_lr / 1e4
    )

    # ── Training Loop ─────────────────────────────────────────────────
    best_auc = 0.0
    best_epoch = 0

    print(f"\n{'Ep':>4}  {'TrLoss':>8}  {'TrAUC':>7}  "
          f"{'VaLoss':>8}  {'VaAUC':>7}  {'LR':>10}")
    print("─" * 58)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_auc = train_epoch(model, tr_ld, optimizer, device, cfg)
        va_loss, va_auc = eval_epoch(model, va_ld, device, cfg)

        # OneCycleLR is stepped per batch, but we already step inside train_epoch
        # (need to step it inside the training loop)
        # NOTE: OneCycleLR steps per BATCH inside train_epoch below,
        # so we don't call scheduler.step() here

        lr_now = optimizer.param_groups[0]["lr"]
        flag = " ✓" if va_auc > best_auc else ""
        print(f"{epoch:>4}  {tr_loss:>8.4f}  {tr_auc:>7.4f}  "
              f"{va_loss:>8.4f}  {va_auc:>7.4f}  {lr_now:>10.2e}{flag}")

        if va_auc > best_auc:
            best_auc = va_auc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "config": cfg,
                "state_dict": model.state_dict(),
                "val_auc": best_auc,
            }, "best_model_v3.pt")

    print("─" * 58)
    print(f"\nBest Val AUC: {best_auc:.4f}  (epoch {best_epoch})")
    print("Model saved → best_model_v3.pt")

    # ── Architecture Diagnostics ──────────────────────────────────────
    print("\n── Learned HetBias (avg over heads, layer 0) ───────────")
    het_bias = model.blocks[0].het.bias.detach().cpu().mean(0)
    type_names = ["target", "user", "item", "act_seq", "cnt_seq", "itm_seq"]
    print("        " + "".join(f"{n:>9}" for n in type_names))
    for i, rn in enumerate(type_names):
        row_str = " ".join(f"{het_bias[i, j].item():+.3f}" for j in range(N_TT))
        print(f"  {rn:<8}  {row_str}")

    print("\n── DIN Position Decay Parameters ────────────────────────")
    r = model.readout
    for name, din in [("action ", r.din_act),
                      ("content", r.din_cnt),
                      ("item   ", r.din_itm)]:
        lam = torch.exp(din.log_decay).item()
        print(f"  {name} seq  λ={lam:.4f}  "
              f"(half-life ≈ {math.log(2) / lam:.1f} positions)")
    print()


# ════════════════════════════════════════════════════════════
# OneCycleLR note: the scheduler must be stepped ONCE PER BATCH.
# We wrap the train_epoch function to handle this correctly.
# ════════════════════════════════════════════════════════════

def train_epoch(  # noqa: F811  (redefinition)
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cfg: Config,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    preds, labels = [], []

    for batch in loader:
        ids = batch["ids"].to(device)
        vals = batch["vals"].to(device)
        types = batch["types"].to(device)
        pos = batch["pos"].to(device)
        mask = batch["mask"].to(device)
        y = batch["labels"].to(device)

        logits = model(ids, vals, types, pos, mask)
        loss = focal_bce_loss(logits, y, cfg.focal_gamma, cfg.label_smooth)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # OneCycleLR: step per batch

        total_loss += loss.item()
        preds.extend(torch.sigmoid(logits).detach().cpu().tolist())
        labels.extend(y.cpu().tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return total_loss / len(loader), auc


def main():  # noqa: F811
    cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")

    print("Loading sample_data.parquet ...")
    df = pd.read_parquet("sample_data.parquet")
    print(f"Shape  : {df.shape}")
    print_diagnostics(df)

    print("Label distribution:")
    cfg.cvr_type = auto_cvr_type(df)

    n = len(df)
    split = int(0.8 * n)
    df_tr = df.iloc[:split].reset_index(drop=True)
    df_va = df.iloc[split:].reset_index(drop=True)
    print(f"\nTrain: {len(df_tr)} | Val: {len(df_va)}")

    pos_tr = df_tr["label"].apply(lambda x: parse_label(x, cfg.cvr_type)).mean()
    pos_va = df_va["label"].apply(lambda x: parse_label(x, cfg.cvr_type)).mean()
    print(f"Train pos_rate: {pos_tr:.3f} | Val pos_rate: {pos_va:.3f}")

    print("\nBuilding train dataset:")
    tr_ds = TAADataset(df_tr, cfg)
    print("\nBuilding val dataset:")
    va_ds = TAADataset(df_va, cfg)

    _col = partial(collate_fn, cfg=cfg)
    tr_ld = DataLoader(tr_ds, cfg.batch_size, shuffle=True,
                       collate_fn=_col, num_workers=2,
                       pin_memory=(device.type == "cuda"))
    va_ld = DataLoader(va_ds, cfg.batch_size, shuffle=False,
                       collate_fn=_col, num_workers=2,
                       pin_memory=(device.type == "cuda"))

    model = UniSFIN(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel       : UniSFIN v3")
    print(f"Parameters  : {n_params:,}")
    print(f"d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, d_ff={cfg.d_ff} (SwiGLU)")
    print(f"Readout: 8 × {cfg.d_model}D  "
          "[target|user_pool|item_pool|din_act|din_cnt|din_itm|fm|cross]")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.wd,
        betas=(0.9, 0.999), eps=1e-8,
    )
    total_steps = cfg.epochs * len(tr_ld)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=total_steps,
        pct_start=cfg.pct_start, anneal_strategy="cos",
        div_factor=25.0, final_div_factor=1e4,
    )

    best_auc = 0.0
    best_epoch = 0

    print(f"\n{'Ep':>4}  {'TrLoss':>8}  {'TrAUC':>7}  "
          f"{'VaLoss':>8}  {'VaAUC':>7}  {'LR':>10}")
    print("─" * 58)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_auc = train_epoch(
            model, tr_ld, optimizer, device, cfg, scheduler)
        va_loss, va_auc = eval_epoch(model, va_ld, device, cfg)

        lr_now = optimizer.param_groups[0]["lr"]
        flag = " ✓" if va_auc > best_auc else ""
        print(f"{epoch:>4}  {tr_loss:>8.4f}  {tr_auc:>7.4f}  "
              f"{va_loss:>8.4f}  {va_auc:>7.4f}  {lr_now:>10.2e}{flag}")

        if va_auc > best_auc:
            best_auc = va_auc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "config": cfg,
                "state_dict": model.state_dict(),
                "val_auc": best_auc,
            }, "best_model_v3.pt")

    print("─" * 58)
    print(f"\nBest Val AUC : {best_auc:.4f}  (epoch {best_epoch})")
    print("Model saved  → best_model_v3.pt")

    print("\n── Learned HetBias (avg over heads, layer 0) ───────────")
    het_bias = model.blocks[0].het.bias.detach().cpu().mean(0)
    type_names = ["target", "user", "item", "act_seq", "cnt_seq", "itm_seq"]
    print("        " + "".join(f"{n:>9}" for n in type_names))
    for i, rn in enumerate(type_names):
        row_str = " ".join(f"{het_bias[i, j].item():+.3f}" for j in range(N_TT))
        print(f"  {rn:<8}  {row_str}")

    print("\n── DIN Position Decay (λ) ───────────────────────────────")
    r = model.readout
    for name, din in [("action ", r.din_act),
                      ("content", r.din_cnt),
                      ("item   ", r.din_itm)]:
        lam = torch.exp(din.log_decay).item()
        print(f"  {name} seq  λ={lam:.4f}  "
              f"(half-life ≈ {math.log(2) / max(lam, 1e-6):.1f} positions)")
    print()


if __name__ == "__main__":
    main()