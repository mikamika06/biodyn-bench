import json
import types
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

ROOT = Path("/Users/macbook/biodyn-bench")
CKPT = ROOT / "data/scgpt_whole_human/scGPT_human"
MASK_VALUE = -1.0
PAD_VALUE = -2.0


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(dev=None, fp16=False):
    from scgpt.model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    vocab = GeneVocab.from_file(CKPT / "vocab.json")
    args = json.load(open(CKPT / "args.json"))
    m = TransformerModel(
        len(vocab), args["embsize"], args["nheads"], args["d_hid"], args["nlayers"],
        vocab=vocab, dropout=args["dropout"], pad_token="<pad>", pad_value=args["pad_value"],
        do_mvc=True, do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False,
        input_emb_style="continuous", n_input_bins=args["n_bins"], cell_emb_style="cls",
        mvc_decoder_style="inner product", ecs_threshold=0.0, explicit_zero_prob=False,
        use_fast_transformer=False, fast_transformer_backend="flash", pre_norm=False,
    )
    sd = torch.load(CKPT / "best_model.pt", map_location="cpu")
    load_pretrained(m, sd, verbose=False)
    m.transformer_encoder.enable_nested_tensor = False
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    m.eval()
    dev = dev or device()
    if fp16:
        m = m.half()
    return m.to(dev), vocab


def load_data(fname="pbmc3k_prepped.h5ad"):
    a = sc.read_h5ad(ROOT / "data/scgpt_collider" / fname)
    genes = list(a.var_names)
    gene_ids = a.var["vocab_id"].to_numpy().astype(np.int64)
    xb = np.asarray(a.layers["X_binned"], dtype=np.float32)
    xl = np.asarray(a.layers["X_log1p"], dtype=np.float32)
    is_tf = a.var["is_tf"].to_numpy()
    return genes, gene_ids, xb, xl, is_tf


class Tokens:
    def __init__(self, gene_ids, vocab, cls=True):
        self.cls = cls
        ids = list(gene_ids)
        if cls:
            ids = [vocab["<cls>"]] + ids
        self.src = torch.tensor(ids, dtype=torch.long)
        self.off = 1 if cls else 0

    def batch(self, xb, dev, dtype=torch.float32):
        n = xb.shape[0]
        vals = torch.tensor(xb, dtype=dtype)
        if self.cls:
            vals = torch.cat([torch.zeros(n, 1, dtype=dtype), vals], 1)
        src = self.src.unsqueeze(0).expand(n, -1).to(dev)
        pad = torch.zeros(n, src.shape[1], dtype=torch.bool, device=dev)
        return src, vals.to(dev), pad


@torch.no_grad()
def forward_mlm(model, src, vals, pad):
    out = model(src, vals, pad, batch_labels=None, CLS=False, CCE=False, MVC=False, ECS=False)
    return out["mlm_output"]


def enable_attention_capture(model, per_head=False, pairs_only=None):
    for layer in model.transformer_encoder.layers:
        if hasattr(layer, "_orig_sa_block"):
            continue
        layer._orig_sa_block = layer._sa_block
        layer._capture = None

        def _sa(self, x, attn_mask, key_padding_mask, is_causal=False):
            o, w = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                                  need_weights=True, average_attn_weights=False, is_causal=is_causal)
            if self._capture is not None:
                self._capture(w)
            return self.dropout1(o)

        layer._sa_block = types.MethodType(_sa, layer)


def set_capture(model, fn):
    for layer in model.transformer_encoder.layers:
        layer._capture = fn


def layers(model):
    return list(model.transformer_encoder.layers)
