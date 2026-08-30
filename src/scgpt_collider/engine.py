import time

import numpy as np
import torch

from scgpt_collider.attn import make_batch
from scgpt_collider.model import MASK_VALUE, enable_attention_capture, forward_mlm, load_data, load_model, set_capture


class Engine:
    def __init__(self, want_attn=True):
        self.genes, self.gene_ids, self.xb, self.xl, self.is_tf = load_data()
        self.model, self.vocab = load_model()
        self.dev = next(self.model.parameters()).device
        self.present = self.xb > 0
        self.want_attn = want_attn
        self.nl = len(self.model.transformer_encoder.layers)
        if want_attn:
            enable_attention_capture(self.model)
        self.cap = []
        set_capture(self.model, (lambda w: self.cap.append(w.mean(1).float().cpu().numpy())) if want_attn else None)

    def donors(self, gene, rng, n=1):
        pool = np.where(self.present[:, gene])[0]
        return rng.choice(pool, n)

    def row(self, cell, read, partner, mask_set, overrides):
        idx = np.where(self.present[cell])[0]
        pos = {g: p for p, g in enumerate(idx)}
        v = self.xb[cell, idx].copy()
        for g, val in overrides.items():
            v[pos[g]] = val
        for g in mask_set:
            v[pos[g]] = MASK_VALUE
        return idx, self.gene_ids[idx], v.astype(np.float32), pos[read], pos[partner]

    def run(self, rows, bs=64, log_every=200):
        order = np.argsort([len(np.where(self.present[r[0]])[0]) for r in rows])
        preds = np.zeros(len(rows), dtype=np.float32)
        att_rp = np.zeros((len(rows), self.nl), dtype=np.float32)
        att_pr = np.zeros((len(rows), self.nl), dtype=np.float32)
        t0 = time.time()
        for bi, s in enumerate(range(0, len(rows), bs)):
            sel = order[s:s + bs]
            items = [self.row(*rows[i]) for i in sel]
            src, vals, pad = make_batch([it[:3] for it in items], self.vocab, self.dev)
            self.cap.clear()
            p = forward_mlm(self.model, src, vals, pad).float().cpu().numpy()
            for r, i in enumerate(sel):
                pr, pp = items[r][3] + 1, items[r][4] + 1
                preds[i] = p[r, pr]
                if self.want_attn:
                    for li in range(self.nl):
                        att_rp[i, li] = self.cap[li][r, pr, pp]
                        att_pr[i, li] = self.cap[li][r, pp, pr]
            if bi % log_every == 0:
                print("batch", bi, "/", len(rows) // bs, "len", src.shape[1], "t", round(time.time() - t0, 1), flush=True)
        return preds, att_rp, att_pr
