"""
Particle Transformer architecture:
  trunc_normal_, pairwise Lorentz helpers, Embed, PairEmbed,
  Block, InstrumentedBlock, SmallParT.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG


# ── Weight initialisation ─────────────────────────────────────────────────────

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


# ── Pairwise Lorentz-invariant feature helpers ───────────────────────────────

@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    return pt2.clamp(min=eps) if eps is not None else pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    return m2.clamp(min=eps) if eps is not None else m2


def atan2(y, x):
    sx       = torch.sign(x)
    sy       = torch.sign(y)
    pi_part  = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt       = torch.sqrt(to_pt2(x, eps=eps))
    rapidity = 0.5 * torch.log(
        1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    m = torch.sqrt(to_m2(x, eps=eps))
    return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    p3     = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2     = p3.square().sum(dim=1, keepdim=True)
    gamma  = (1 - b2).clamp(min=eps) ** (-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v  = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(
        xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(
        xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta   = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin   = torch.minimum(pti, ptj)
        lnkt    = torch.log((ptmin * delta).clamp(min=eps))
        lnz     = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij  = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    if num_outputs > 5:
        xj_boost  = boost(xj, xi + xj)
        costheta  = (p3_norm(xj_boost, eps=eps) *
                     p3_norm(xi + xj,  eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert len(outputs) == num_outputs
    return torch.cat(outputs, dim=1)


# ── Embed ─────────────────────────────────────────────────────────────────────

class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation="gelu"):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list   = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        # x: (batch, input_dim, seq_len)
        if self.input_bn is not None:
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # returns (seq_len, batch, embed_dim)
        return self.embed(x)


# ── PairEmbed ─────────────────────────────────────────────────────────────────

class PairEmbed(nn.Module):
    def __init__(self, pairwise_lv_dim, pairwise_input_dim, dims,
                 remove_self_pair=False, use_pre_activation_pair=True,
                 mode="sum", normalize_input=True, activation="gelu",
                 eps=1e-8, for_onnx=False):
        super().__init__()
        self.pairwise_lv_dim    = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric       = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair   = remove_self_pair
        self.mode               = mode
        self.for_onnx           = for_onnx
        self.pairwise_lv_fts    = partial(pairwise_lv_fts,
                                          num_outputs=pairwise_lv_dim,
                                          eps=eps, for_onnx=for_onnx)
        self.out_dim = dims[-1]

        if self.mode == "sum":
            if pairwise_lv_dim > 0:
                input_dim   = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

    def forward(self, x, uu=None):
        assert x is not None
        with torch.no_grad():
            batch_size, _, seq_len = x.size()
            if self.is_symmetric:
                i, j = torch.tril_indices(seq_len, seq_len,
                                           offset=0, device=x.device)
                x_   = x.unsqueeze(-1).expand(-1, -1, -1, seq_len)
                xi   = x_[:, :, i, j]
                xj   = x_[:, :, j, i]
                x_   = self.pairwise_lv_fts(xi, xj)
            else:
                x_   = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                x_   = x_.view(-1, self.pairwise_lv_dim, seq_len * seq_len)

        elements = self.embed(x_)

        if self.is_symmetric:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len,
                            dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y    # (batch, num_heads, P, P)


# ── Block ─────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation="gelu",
                 scale_fc=True, scale_attn=True,
                 scale_heads=True, scale_resids=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.ffn_dim   = embed_dim * ffn_ratio

        self.pre_attn_norm  = nn.LayerNorm(embed_dim)
        self.attn           = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, add_bias_kv=add_bias_kv)
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn  else None
        self.dropout        = nn.Dropout(dropout)

        self.pre_fc_norm  = nn.LayerNorm(embed_dim)
        self.fc1          = nn.Linear(embed_dim, self.ffn_dim)
        self.act          = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.act_dropout  = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2          = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn  = nn.Parameter(torch.ones(num_heads), requires_grad=True) \
                       if scale_heads  else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) \
                       if scale_resids else None

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        if x_cls is not None:
            with torch.no_grad():
                cls_pad      = torch.zeros(padding_mask.size(0), 1,
                                           dtype=padding_mask.dtype,
                                           device=padding_mask.device)
                padding_mask = torch.cat((cls_pad, padding_mask), dim=1)
            residual = x_cls
            u        = torch.cat((x_cls, x), dim=0)
            u        = self.pre_attn_norm(u)
            x        = self.attn(x_cls, u, u,
                                 key_padding_mask=padding_mask,
                                 need_weights=False)[0]
        else:
            residual = x
            x        = self.pre_attn_norm(x)
            x        = self.attn(x, x, x,
                                 key_padding_mask=None,
                                 attn_mask=attn_mask,
                                 need_weights=False)[0]

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum("tbhd,h->tbdh", x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x        = self.pre_fc_norm(x)
        x        = self.act(self.fc1(x))
        x        = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual
        return x


# ── InstrumentedBlock ─────────────────────────────────────────────────────────

class InstrumentedBlock(Block):
    """
    Identical to Block but saves per-head attention weights after
    every forward pass.
      last_attn_weights_per_head : (batch, num_heads, seq, seq)  on CPU
    """
    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        if x_cls is not None:
            with torch.no_grad():
                cls_pad      = torch.zeros(padding_mask.size(0), 1,
                                           dtype=padding_mask.dtype,
                                           device=padding_mask.device)
                padding_mask = torch.cat((cls_pad, padding_mask), dim=1)
            residual = x_cls
            u        = torch.cat((x_cls, x), dim=0)
            u        = self.pre_attn_norm(u)
            attn_out, attn_w = self.attn(
                x_cls, u, u,
                key_padding_mask=padding_mask,
                need_weights=True,
                average_attn_weights=False)
            x = attn_out
        else:
            residual = x
            x_norm   = self.pre_attn_norm(x)
            attn_out, attn_w = self.attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=None,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False)
            x = attn_out

        self.last_attn_weights_per_head = attn_w.detach().cpu()

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum("tbhd,h->tbdh", x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x        = self.pre_fc_norm(x)
        x        = self.act(self.fc1(x))
        x        = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual
        return x


# ── SmallParT ─────────────────────────────────────────────────────────────────

class SmallParT(nn.Module):
    """
    4-layer / 4-head Particle Transformer with full instrumentation.

    After every forward() call:
      residual_stream  : list[tensor(batch, P, 128)]    length = num_layers + 1
      attn_weights     : list[tensor(batch, heads, P, P)]  length = num_layers
      cls_attn_weights : list[tensor(batch, heads, 1, P+1)] length = num_cls_layers
    """

    def __init__(self, cfg):
        super().__init__()
        C         = cfg
        embed_dim = C["embed_dim"]
        num_heads = C["num_heads"]

        self.num_heads = num_heads

        self.embed = Embed(input_dim=7, dims=[128, 512, 128], activation="gelu")

        self.pair_embed = PairEmbed(
            pairwise_lv_dim    = C["pair_input_dim"],
            pairwise_input_dim = 0,
            dims               = C["pair_embed_dims"] + [num_heads],
            remove_self_pair   = False,
            use_pre_activation_pair = True,
            for_onnx           = False)

        block_cfg = dict(
            embed_dim=embed_dim, num_heads=num_heads,
            ffn_ratio=C["ffn_ratio"],
            dropout=C["dropout"], attn_dropout=C["dropout"],
            activation_dropout=C["dropout"],
            add_bias_kv=False, activation="gelu",
            scale_fc=True, scale_attn=True,
            scale_heads=True, scale_resids=True)

        cls_cfg = {**block_cfg,
                   "dropout": 0, "attn_dropout": 0, "activation_dropout": 0}

        self.blocks     = nn.ModuleList(
            [InstrumentedBlock(**block_cfg) for _ in range(C["num_layers"])])
        self.cls_blocks = nn.ModuleList(
            [InstrumentedBlock(**cls_cfg)   for _ in range(C["num_cls_layers"])])

        self.norm      = nn.LayerNorm(embed_dim)
        self.fc        = nn.Linear(embed_dim, C["num_classes"])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        self.residual_stream  = []
        self.attn_weights     = []
        self.cls_attn_weights = []

    def forward(self, x, v, mask):
        """
        x    : (batch, 7, P)
        v    : (batch, 4, P)  px, py, pz, E
        mask : (batch, 1, P)  float  1 = real  0 = padding
        """
        self.residual_stream  = []
        self.attn_weights     = []
        self.cls_attn_weights = []

        batch_size = x.size(0)
        P          = x.size(2)
        num_heads  = self.num_heads

        real_mask = mask.bool().squeeze(1)             # (batch, P)
        pad_mask  = ~real_mask                         # (batch, P)

        pair_bias  = self.pair_embed(v)                # (batch, heads, P, P)
        col_mask   = (pad_mask.float() * -1e4)[:, None, None, :]
        row_mask   = (pad_mask.float() * -1e4)[:, None, :, None]
        attn_bias  = pair_bias + col_mask + row_mask
        attn_mask  = attn_bias.reshape(batch_size * num_heads, P, P)

        x = self.embed(x)
        x = x.masked_fill(pad_mask.T.unsqueeze(-1), 0.)
        self.residual_stream.append(x.permute(1, 0, 2).detach().cpu())

        for block in self.blocks:
            x = block(x, x_cls=None, padding_mask=None, attn_mask=attn_mask)
            x = x.masked_fill(pad_mask.T.unsqueeze(-1), 0.)
            self.residual_stream.append(x.permute(1, 0, 2).detach().cpu())
            self.attn_weights.append(block.last_attn_weights_per_head)

        cls_tokens = self.cls_token.expand(1, batch_size, -1).clone()
        for cls_block in self.cls_blocks:
            cls_tokens = cls_block(x, x_cls=cls_tokens, padding_mask=pad_mask)
            self.cls_attn_weights.append(cls_block.last_attn_weights_per_head)

        x_cls = self.norm(cls_tokens).squeeze(0)
        return self.fc(x_cls)
