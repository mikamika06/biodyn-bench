import resource
import time

import numpy as np
import torch

from scgpt_collider.attn import make_batch
from scgpt_collider.model import MASK_VALUE, enable_attention_capture, forward_mlm, load_data, load_model, set_capture


class Engine:
    def __init__(self, want_attn=True, fname="pbmc3k_prepped.h5ad"):
        self.genes, self.gene_ids, self.xb, self.xl, self.is_tf = load_data(fname)
        self.model, self.vocab = load_model()
        self.dev = next(self.model.parameters()).device
        self.present = self.xb > 0
        self.want_attn = want_attn
        self.nl = len(self.model.transformer_encoder.layers)
        if want_attn:
            enable_attention_capture(self.model)
        self.cap = []
        self._ar = self._pa = self._pb = None

        def _cap(w):
            m = w.mean(1).float()
            self.cap.append(torch.stack([m[self._ar, self._pa, self._pb], m[self._ar, self._pb, self._pa]]).cpu().numpy())

        set_capture(self.model, _cap if want_attn else None)

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

    def run(self, rows, bs=64, log_every=200, budget=6e6):
        lens = np.array([int(self.present[r[0]].sum()) for r in rows])
        order = np.argsort(lens)
        batches = []
        cur = []
        for i in order:
            L = lens[i] + 1
            if cur and (len(cur) + 1) * L * L > budget or len(cur) >= bs:
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        preds = np.zeros(len(rows), dtype=np.float32)
        att_rp = np.zeros((len(rows), self.nl), dtype=np.float32)
        att_pr = np.zeros((len(rows), self.nl), dtype=np.float32)
        t0 = time.time()
        for bi, sel in enumerate(batches):
            items = [self.row(*rows[i]) for i in sel]
            src, vals, pad = make_batch([it[:3] for it in items], self.vocab, self.dev)
            if self.want_attn:
                self._ar = torch.arange(len(sel), device=self.dev)
                self._pa = torch.tensor([it[3] + 1 for it in items], device=self.dev)
                self._pb = torch.tensor([it[4] + 1 for it in items], device=self.dev)
            self.cap.clear()
            p = forward_mlm(self.model, src, vals, pad).float().cpu().numpy()
            for r, i in enumerate(sel):
                preds[i] = p[r, items[r][3] + 1]
                if self.want_attn:
                    for li in range(self.nl):
                        att_rp[i, li] = self.cap[li][0, r]
                        att_pr[i, li] = self.cap[li][1, r]
            if self.dev.type == "mps" and bi % 25 == 0:
                torch.mps.empty_cache()
            if bi % log_every == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
                print("batch", bi, "/", len(batches), "n", len(sel), "len", src.shape[1], "t", round(time.time() - t0, 1), "rss", round(rss, 1), flush=True)
        return preds, att_rp, att_pr
