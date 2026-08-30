import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/macbook/biodyn-bench/src")
from scgpt_collider.model import CKPT, Tokens, forward_mlm, load_data, load_model

genes, gene_ids, xb, xl, is_tf = load_data()
m, vocab = load_model(dev=torch.device("cpu"))
sd = torch.load(CKPT / "best_model.pt", map_location="cpu")
sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
md = m.state_dict()
same = sum(1 for k in sd if k in md and torch.equal(md[k], sd[k]))
print("ckpt keys", len(sd), "model keys", len(md), "identical", same, "not in model", [k for k in sd if k not in md][:5], "not in ckpt", [k for k in md if k not in sd][:5])
tok = Tokens(gene_ids, vocab)
src, vals, pad = tok.batch(xb[:4], torch.device("cpu"))
p = forward_mlm(m, src, vals, pad)
print("cpu pred visible mean", float(p[:, 1:].mean()))
rng = np.random.default_rng(0)
for name, keep_zero in (("all genes", True), ("nonzero only", False)):
    for c in range(2):
        v = xb[c]
        idx = np.arange(len(v)) if keep_zero else np.where(v > 0)[0]
        ids = torch.tensor([vocab["<cls>"]] + list(gene_ids[idx])).unsqueeze(0)
        vv = torch.tensor(np.r_[0.0, v[idx]], dtype=torch.float32).unsqueeze(0)
        pd_ = torch.zeros_like(ids, dtype=torch.bool)
        mpos = rng.choice(np.arange(1, ids.shape[1]), max(1, int(0.25 * (ids.shape[1] - 1))), replace=False)
        vm = vv.clone(); vm[0, mpos] = -1.0
        pr = forward_mlm(m, ids, vm, pd_)[0, mpos]
        tr = vv[0, mpos]
        print(name, "cell", c, "len", ids.shape[1], "mae", float((pr - tr).abs().mean()), "const-mae", float((tr - tr.mean()).abs().mean()), "corr", float(np.corrcoef(pr.numpy(), tr.numpy())[0, 1]), "pred mean", float(pr.mean()), "true mean", float(tr.mean()))
