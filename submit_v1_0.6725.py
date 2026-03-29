#!/usr/bin/env python3
"""
Unified Recommender: Heterogeneous Transformer for CVR Prediction
TAAC2026 Competition Baseline

Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │  Feature-as-Token Tokenization                           │
  │  [target] | [user fields] | [item fields]                │
  │         | [action_seq] | [content_seq] | [item_seq]      │
  ├──────────────────────────────────────────────────────────┤
  │  N × HeterogeneousTransformerLayer                       │
  │    - Standard MHA + FFN (Pre-LayerNorm)                  │
  │    - + Learnable type-pair attention bias B[h,ti,tj]     │
  │      controlling cross-paradigm attention patterns       │
  ├──────────────────────────────────────────────────────────┤
  │  Target Token Representation → MLP → CVR logit           │
  └──────────────────────────────────────────────────────────┘

Token Types:
  0: target item token       (the item to score)
  1: user field tokens       (non-sequential user features)
  2: item field tokens       (non-sequential item features)
  3: action sequence tokens  (seq_feature.action_seq)
  4: content sequence tokens (seq_feature.content_seq)
  5: item sequence tokens    (seq_feature.item_seq)
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

TT_TARGET   = 0  # target item token
TT_USER     = 1  # user non-sequential field tokens
TT_ITEM     = 2  # item non-sequential field tokens
TT_ACT_SEQ  = 3  # action_seq tokens
TT_CONT_SEQ = 4  # content_seq tokens
TT_ITEM_SEQ = 5  # item_seq tokens
N_TT        = 6  # total token types


@dataclass
class Config:
    # ── Embedding ──────────────────────────────────────
    d_model: int      = 64
    vocab_size: int   = 4096   # feature-ID + item-ID vocabulary size
    max_seq_len: int  = 50     # max tokens per sequence sub-type
    max_field_t: int  = 48     # max non-sequential field tokens per side

    # ── Transformer ────────────────────────────────────
    n_layers: int  = 3
    n_heads: int   = 4
    d_ff: int      = 256
    dropout: float = 0.1000

    # ── Training ───────────────────────────────────────
    batch_size: int = 128
    lr: float       = 1.00e-03
    wd: float       = 1.00e-04
    epochs: int     = 60
    grad_clip: float = 1.0

    # ── Label ──────────────────────────────────────────
    cvr_type: int = 1  # min action_type counted as conversion (auto-tuned)


# ════════════════════════════════════════════════════════════
# 1. Data Parsing
# ════════════════════════════════════════════════════════════

def _get(obj, key, default=None):
    """统一字典/结构体字段访问。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_val(v: float) -> float:
    """
    Map raw feature value to a compact scale.
    Uses sign-preserving log1p to handle large integers gracefully.
    """
    v = float(v)
    return math.log1p(abs(v)) * (1.0 if v >= 0 else -1.0)


def parse_label(label_arr, cvr_type: int = 1) -> int:
    """Binary CVR label: 1 iff any action with action_type >= cvr_type exists."""
    if label_arr is None:
        return 0
    for a in label_arr:
        if _get(a, "action_type", 0) >= cvr_type:
            return 1
    return 0


def parse_feature_array(
    feat_arr, max_tokens: int
) -> Tuple[List[int], List[float]]:
    """
    Parse an array of feature structs → (feature_ids, normalized_values).

    Supported feature_value_type:
      int_value   → single int
      float_value → single float
      int_array   → up to 8 array elements
      float_array → up to 8 array elements
      combinations of the above
    """
    fids: List[int]   = []
    fvals: List[float] = []

    if feat_arr is None:
        return fids, fvals

    for feat in feat_arr:
        fid   = int(_get(feat, "feature_id", 0)) % (4095) + 1   # [1, 4095]
        ftype = _get(feat, "feature_value_type", "") or ""

        def emit(v):
            fids.append(fid)
            fvals.append(normalize_val(v))

        # scalar int
        if "int_value" in ftype and "array" not in ftype:
            v = _get(feat, "int_value", None)
            if v is not None:
                emit(v)

        # scalar float
        if ftype == "float_value":
            v = _get(feat, "float_value", None)
            if v is not None:
                emit(v)

        # int array  (up to 8 values per feature)
        if "int_array" in ftype:
            arr = _get(feat, "int_array", None)
            if arr is not None:
                for x in arr[:8]:
                    emit(x)

        # float array (up to 8 values per feature)
        if "float_array" in ftype:
            arr = _get(feat, "float_array", None)
            if arr is not None:
                for x in arr[:8]:
                    emit(x)

        if len(fids) >= max_tokens:
            break

    return fids[:max_tokens], fvals[:max_tokens]


def parse_seq_sub(
    seq_feature, sub_key: str, max_seq_len: int
) -> Tuple[List[int], List[int]]:
    """
    Parse one sequence sub-type (action_seq / content_seq / item_seq).

    Schema: seq_feature[sub_key] is an array of feature structs,
    each with int_array = [v_t0, v_t1, v_t2, ...] — one value per timestep.
    We use the first feature struct as the primary sequence token IDs.

    Returns (token_ids, positions) where positions are 1-indexed.
    """
    if seq_feature is None:
        return [], []

    sub = _get(seq_feature, sub_key, None)
    if sub is None or len(sub) == 0:
        return [], []

    tok_ids: List[int] = []
    positions: List[int] = []

    for feat in sub:
        arr = _get(feat, "int_array", None)
        if arr is not None:
            for pos, v in enumerate(arr[:max_seq_len]):
                tok_ids.append(int(v) % 4095 + 1)  # map to [1, 4095]
                positions.append(pos + 1)            # 1-indexed
            break  # use first feature struct as primary sequence

    return tok_ids[:max_seq_len], positions[:max_seq_len]


# ════════════════════════════════════════════════════════════
# 2. Dataset & Collate
# ════════════════════════════════════════════════════════════

class TAADataset(Dataset):
    """
    Pre-processes each row into a sample dict of token lists.
    Heavy parsing is done once in __init__ (not in __getitem__).
    """

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
        target_id = int(row["item_id"]) % 4095 + 1

        u_fids, u_fvals = parse_feature_array(row["user_feature"], cfg.max_field_t)
        i_fids, i_fvals = parse_feature_array(row["item_feature"], cfg.max_field_t)

        seq = row["seq_feature"]
        act_ids, act_pos = parse_seq_sub(seq, "action_seq",  cfg.max_seq_len)
        cnt_ids, cnt_pos = parse_seq_sub(seq, "content_seq", cfg.max_seq_len)
        itm_ids, itm_pos = parse_seq_sub(seq, "item_seq",    cfg.max_seq_len)

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
    """
    Assemble a batch into padded tensors.

    Unified token sequence layout per sample:
      [target] | [user_fields] | [item_fields]
              | [action_seq]  | [content_seq] | [item_seq]

    Outputs (all shape B × T_max):
      ids   — feature / item IDs (long)
      vals  — normalized scalar values (float)
      types — token type [0–5] (long)
      pos   — position within sequence, 0 = non-sequential (long)
      mask  — True = valid token (bool)
    """
    B = len(batch)
    all_ids, all_vals, all_types, all_pos = [], [], [], []

    for s in batch:
        ids, vals, types, pos = [], [], [], []

        def add(i, v, t, p):
            ids.append(i); vals.append(v); types.append(t); pos.append(p)

        # ── target token ──────────────────────────────────────
        add(s["target_id"], 1.0, TT_TARGET, 0)

        # ── user field tokens ─────────────────────────────────
        for fid, fv in zip(s["u_fids"], s["u_fvals"]):
            add(fid, fv, TT_USER, 0)

        # ── item field tokens ─────────────────────────────────
        for fid, fv in zip(s["i_fids"], s["i_fvals"]):
            add(fid, fv, TT_ITEM, 0)

        # ── sequence tokens ───────────────────────────────────
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
    Unified Feature-as-Token embedding:

      emb(x) = FeatureIDEmb(id) ⊙ ValueGate(v)   ← content
             + TokenTypeEmb(type)                  ← paradigm signal
             + PositionEmb(pos)                    ← temporal order (seq only)

    ValueGate: a sigmoid MLP that gates the feature embedding based on
    the scalar value, so identical feature IDs with different values
    produce distinct representations.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model

        self.feat_emb = nn.Embedding(cfg.vocab_size, D, padding_idx=0)
        self.type_emb = nn.Embedding(N_TT, D)

        # +2: slot 0 = "no position" (field tokens), slots 1..max_seq_len = seq positions
        self.pos_emb  = nn.Embedding(cfg.max_seq_len + 2, D)

        # Scalar value → soft gate  (B, T) → (B, T, D)
        self.val_gate = nn.Sequential(
            nn.Linear(1, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D), nn.Sigmoid(),
        )

        self.norm    = nn.LayerNorm(D)
        self.dropout = nn.Dropout(cfg.dropout)
        self._init()

    def _init(self):
        nn.init.normal_(self.feat_emb.weight, std=0.01)
        nn.init.normal_(self.type_emb.weight, std=0.01)
        nn.init.normal_(self.pos_emb.weight,  std=0.01)

    def forward(
        self,
        ids:   torch.Tensor,   # (B, T) long
        vals:  torch.Tensor,   # (B, T) float
        types: torch.Tensor,   # (B, T) long
        pos:   torch.Tensor,   # (B, T) long
    ) -> torch.Tensor:         # (B, T, D)

        # Clamp positions to valid embedding range
        pos = pos.clamp(0, self.pos_emb.num_embeddings - 1)

        gate = self.val_gate(vals.unsqueeze(-1))     # (B, T, D) ∈ (0,1)
        x    = self.feat_emb(ids) * gate             # value-gated feature embedding
        x    = x + self.type_emb(types)              # + token type
        x    = x + self.pos_emb(pos)                 # + sequence position
        return self.dropout(self.norm(x))


class HetBias(nn.Module):
    """
    Heterogeneous attention bias: a learnable n_heads × N_TT × N_TT matrix.

    result[b, h, i, j] = B[h, type_q[b,i], type_k[b,j]]

    Added to attention logits before softmax, this lets the model learn
    how much each token type pair should attend to each other by default,
    bridging the sequential and non-sequential paradigms.
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.H    = n_heads
        self.bias = nn.Parameter(torch.zeros(n_heads, N_TT, N_TT))

    def forward(
        self, tq: torch.Tensor, tk: torch.Tensor
    ) -> torch.Tensor:
        """
        tq: (B, Tq) — query token types
        tk: (B, Tk) — key   token types
        →   (B, H, Tq, Tk)
        """
        B, Tq = tq.shape
        Tk    = tk.shape[1]
        H     = self.H

        # Expand to (B*H, Tq) and (B*H, Tk)
        tq_bh = tq.unsqueeze(1).expand(B, H, Tq).reshape(B * H, Tq)
        tk_bh = tk.unsqueeze(1).expand(B, H, Tk).reshape(B * H, Tk)

        # Expand bias to (B*H, N_TT, N_TT)
        bias = (self.bias
                .unsqueeze(0)
                .expand(B, H, N_TT, N_TT)
                .reshape(B * H, N_TT, N_TT))

        # Gather row corresponding to each query type: → (B*H, Tq, N_TT)
        row_idx = tq_bh.unsqueeze(-1).expand(B * H, Tq, N_TT)
        rows    = bias.gather(1, row_idx)

        # Gather col corresponding to each key type: → (B*H, Tq, Tk)
        col_idx = tk_bh.unsqueeze(1).expand(B * H, Tq, Tk)
        result  = rows.gather(2, col_idx)

        return result.reshape(B, H, Tq, Tk)


class HetTransformerLayer(nn.Module):
    """
    Single stackable Transformer layer with heterogeneous attention bias.

    Design choices:
    - Pre-LayerNorm (more stable training)
    - Fused QKV projection
    - Heterogeneous bias added to raw attention logits
    - GELU FFN
    """

    def __init__(self, cfg: Config):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, "d_model must be divisible by n_heads"
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
        self.n1 = nn.LayerNorm(D)
        self.n2 = nn.LayerNorm(D)
        self.dp = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x:     torch.Tensor,            # (B, T, D)
        types: torch.Tensor,            # (B, T) int
        mask:  Optional[torch.Tensor],  # (B, T) bool — True = valid
    ) -> torch.Tensor:                  # (B, T, D)

        B, T, D = x.shape
        H, Dh   = self.H, self.Dh

        # ── Multi-Head Self-Attention (Pre-Norm) ─────────────
        residual = x
        x = self.n1(x)

        Q, K, V = self.qkv(x).split(D, dim=-1)
        def split_heads(t):
            return t.view(B, T, H, Dh).transpose(1, 2)   # (B, H, T, Dh)
        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product + heterogeneous bias
        logits  = (Q @ K.transpose(-2, -1)) / math.sqrt(Dh)  # (B, H, T, T)
        logits += self.het(types, types)                       # + B[h, ti, tj]

        # Mask padding keys (True=valid → negate for mask_fill)
        if mask is not None:
            logits = logits.masked_fill(
                ~mask[:, None, None, :], float("-inf")
            )

        w = F.softmax(logits, dim=-1)
        w = torch.nan_to_num(w)     # guard all-masked rows
        w = self.dp(w)

        attn = (w @ V).transpose(1, 2).reshape(B, T, D)
        x = residual + self.dp(self.out(attn))

        # ── Feed-Forward (Pre-Norm) ───────────────────────────
        x = x + self.dp(self.ffn(self.n2(x)))
        return x


class UnifiedRecommender(nn.Module):
    """
    Full model:
      TokenEmbedding
        → N × HeterogeneousTransformerLayer
          → LayerNorm
            → Target Token [0] representation
              → MLP prediction head
                → CVR logit (scalar per sample)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.emb    = TokenEmbedding(cfg)
        self.layers = nn.ModuleList(
            [HetTransformerLayer(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm   = nn.LayerNorm(cfg.d_model)
        self.head   = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(
        self,
        ids:   torch.Tensor,  # (B, T)
        vals:  torch.Tensor,  # (B, T)
        types: torch.Tensor,  # (B, T)
        pos:   torch.Tensor,  # (B, T)
        mask:  torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:        # (B,) logits

        x = self.emb(ids, vals, types, pos)         # (B, T, D)
        for layer in self.layers:
            x = layer(x, types, mask)
        x      = self.norm(x)
        target = x[:, 0]                            # target token is always slot 0
        return self.head(target).squeeze(-1)        # (B,)


# ════════════════════════════════════════════════════════════
# 4. Training & Evaluation
# ════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Tuple[float, float]:
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
        loss   = F.binary_cross_entropy_with_logits(logits, y)

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
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
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
        loss   = F.binary_cross_entropy_with_logits(logits, y)

        total_loss += loss.item()
        preds.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(y.cpu().tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    return total_loss / len(loader), auc


# ════════════════════════════════════════════════════════════
# 5. Main
# ════════════════════════════════════════════════════════════

def auto_cvr_type(df: pd.DataFrame) -> int:
    """
    Auto-select the minimum action_type threshold that gives a non-degenerate
    positive rate (between 1% and 99%).
    """
    for t in [1, 2, 3]:
        pos_rate = df["label"].apply(lambda x: parse_label(x, t)).mean()
        print(f"  action_type >= {t}: pos_rate = {pos_rate:.3f}")
        if 0.01 <= pos_rate <= 0.99:
            print(f"  → Using cvr_type = {t}  (pos_rate={pos_rate:.3f})")
            return t
    print("  Warning: could not find non-degenerate label, using cvr_type=1")
    return 1


def print_data_diagnostics(df: pd.DataFrame):
    """Print a quick diagnostic of the dataset."""
    print("\n── Dataset Diagnostics ───────────────────────────────")
    print(f"  Rows: {len(df)}")

    # Sample one row to understand structure
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

    # Label peek
    lbl = row["label"]
    print(f"  label sample        : {list(lbl)[:3]}")
    print("──────────────────────────────────────────────────────\n")


def main():
    cfg = Config(
        d_model=64, n_layers=3, n_heads=4, d_ff=256, dropout=0.1000,
        vocab_size=4096, max_seq_len=50, max_field_t=48,
        batch_size=128, lr=1.00e-03, wd=1.00e-04, epochs=60, grad_clip=1.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")

    # ── Load ──────────────────────────────────────────────────────────
    print("Loading sample_data.parquet ...")
    df = pd.read_parquet("sample_data.parquet")
    print(f"Shape  : {df.shape}")

    print_data_diagnostics(df)

    # ── Auto-select CVR label threshold ──────────────────────────────
    print("Label distribution:")
    cfg.cvr_type = auto_cvr_type(df)

    # ── Train / Val split (80/20, time-ordered) ───────────────────────
    n     = len(df)
    split = int(0.8 * n)
    df_tr = df.iloc[:split].reset_index(drop=True)
    df_va = df.iloc[split:].reset_index(drop=True)
    print(f"\nTrain: {len(df_tr)} samples | Val: {len(df_va)} samples")

    pos_tr = df_tr["label"].apply(lambda x: parse_label(x, cfg.cvr_type)).mean()
    pos_va = df_va["label"].apply(lambda x: parse_label(x, cfg.cvr_type)).mean()
    print(f"Train pos_rate: {pos_tr:.3f} | Val pos_rate: {pos_va:.3f}")

    # ── Datasets & DataLoaders ────────────────────────────────────────
    print("\nBuilding datasets:")
    tr_ds = TAADataset(df_tr, cfg)
    print("\nBuilding datasets:")
    va_ds = TAADataset(df_va, cfg)

    _col  = partial(collate_fn, cfg=cfg)
    tr_ld = DataLoader(tr_ds, cfg.batch_size, shuffle=True,
                       collate_fn=_col, num_workers=0, pin_memory=(device.type=="cuda"))
    va_ld = DataLoader(va_ds, cfg.batch_size, shuffle=False,
                       collate_fn=_col, num_workers=0, pin_memory=(device.type=="cuda"))

    # ── Model ─────────────────────────────────────────────────────────
    model    = UnifiedRecommender(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters : {n_params:,}")

    # Print architecture summary
    print("\nArchitecture:")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, d_ff={cfg.d_ff}")
    print(f"  vocab_size={cfg.vocab_size}, max_seq_len={cfg.max_seq_len}")
    print(f"  HetBias matrix: {cfg.n_heads} heads × {N_TT}×{N_TT} type pairs")

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.wd
    )
    # Cosine decay with warm-up via linear LR scaling first 2 epochs
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=2
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs - 2, eta_min=cfg.lr * 0.01
            ),
        ],
        milestones=[2],
    )

    # ── Training Loop ─────────────────────────────────────────────────
    best_auc  = 0.0
    best_epoch = 0

    print(f"\n{'Epoch':>6}  {'Tr Loss':>8}  {'Tr AUC':>8}  {'Va Loss':>8}  {'Va AUC':>8}  {'LR':>10}")
    print("─" * 62)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_auc = train_epoch(
            model, tr_ld, optimizer, device, cfg.grad_clip
        )
        va_loss, va_auc = eval_epoch(model, va_ld, device)
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
                "best_model_v1.pt",
            )

    print("─" * 62)
    print(f"\nBest Val AUC: {best_auc:.4f}  (epoch {best_epoch})")
    print("Model saved to best_model.pt")

    # ── Final Diagnostics ─────────────────────────────────────────────
    print("\n── Learned HetBias (averaged over heads) ────────────────")
    het_bias = model.layers[0].het.bias.detach().cpu().mean(0)  # (N_TT, N_TT)
    type_names = ["target", "user", "item", "act_seq", "cnt_seq", "itm_seq"]
    header = "        " + "".join(f"{n:>9}" for n in type_names)
    print(header)
    for i, row_name in enumerate(type_names):
        row_str = " ".join(f"{het_bias[i, j].item():+.3f}" for j in range(N_TT))
        print(f"  {row_name:<8}  {row_str}")
    print()


if __name__ == "__main__":
    main()