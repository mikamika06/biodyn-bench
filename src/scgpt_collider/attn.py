import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/Users/macbook/biodyn-bench/src")
from scgpt_collider.model import MASK_VALUE, PAD_VALUE, enable_attention_capture, forward_mlm, load_data, load_model, set_capture

OUT = Path("/Users/macbook/biodyn-bench/out/scgpt_collider")


def cell_tokens(xb, gene_ids, vocab, c):
    idx = np.where(xb[c] > 0)[0]
    return idx, gene_ids[idx], xb[c, idx]


def make_batch(items, vocab, dev):
    L = max(len(it[1]) for it in items) + 1
    n = len(items)
    src = torch.full((n, L), vocab["<pad>"], dtype=torch.long)
    vals = torch.full((n, L), PAD_VALUE, dtype=torch.float32)
    pad = torch.ones(n, L, dtype=torch.bool)
    for r, (idx, ids, v) in enumerate(items):
        k = len(ids)
        src[r, 0] = vocab["<cls>"]
        vals[r, 0] = 0.0
        src[r, 1:k + 1] = torch.from_numpy(ids)
        vals[r, 1:k + 1] = torch.from_numpy(v)
        pad[r, :k + 1] = False
    return src.to(dev), vals.to(dev), pad.to(dev)


def run(n_cells=None, bs=32, seed=0):
    genes, gene_ids, xb, xl, is_tf = load_data()
    m, vocab = load_model()
    dev = next(m.parameters()).device
    G = len(genes)
    nl = len(m.transformer_encoder.layers)
    n_cells = n_cells or xb.shape[0]
    cells = list(range(n_cells))
    lens = [(xb[c] > 0).sum() for c in cells]
    order = np.argsort(lens)
    enable_attention_capture(m)
    ssum = np.zeros((nl, G, G), dtype=np.float64)
    ssum_len = np.zeros((nl, G, G), dtype=np.float64)
    cnt = np.zeros((G, G), dtype=np.int32)
    cap = []
    set_capture(m, lambda w: cap.append(w.mean(1).float().cpu().numpy()))
    t0 = time.time()
    for s in range(0, n_cells, bs):
        b = [cells[i] for i in order[s:s + bs]]
        items = [cell_tokens(xb, gene_ids, vocab, c) for c in b]
        src, vals, pad = make_batch(items, vocab, dev)
        cap.clear()
        forward_mlm(m, src, vals, pad)
        for r, (idx, ids, v) in enumerate(items):
            k = len(idx)
            ix = np.ix_(idx, idx)
            for li in range(nl):
                w = cap[li][r, 1:k + 1, 1:k + 1]
                ssum[li][ix] += w
                ssum_len[li][ix] += w * (k + 1)
            cnt[ix] += 1
        if (s // bs) % 10 == 0:
            print("batch", s // bs, "len", src.shape[1], "t", round(time.time() - t0, 1), flush=True)
    den = np.maximum(cnt, 1)[None]
    attn_layers = (ssum / den).astype(np.float32)
    attn_len = (ssum_len / den).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "attn_layers.npy", attn_layers)
    np.save(OUT / "attn_layers_lennorm.npy", attn_len)
    np.save(OUT / "attn_mean.npy", attn_layers.mean(0))
    np.save(OUT / "attn_count.npy", cnt)
    open(OUT / "genes.txt", "w").write("\n".join(genes))
    json.dump({"n_cells": n_cells, "n_genes": G, "n_layers": nl, "seconds": time.time() - t0, "median_tokens": float(np.median(lens)), "max_tokens": int(max(lens)), "device": str(dev)}, open(OUT / "attn_info.json", "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else None)
