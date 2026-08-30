import sys
import time

import torch

sys.path.insert(0, "/Users/macbook/biodyn-bench/src")
from scgpt_collider.model import Tokens, enable_attention_capture, forward_mlm, load_data, load_model, set_capture

genes, gene_ids, xb, xl, is_tf = load_data()
m, vocab = load_model()
dev = next(m.parameters()).device
tok = Tokens(gene_ids, vocab)
for bs in (8, 32):
    src, vals, pad = tok.batch(xb[:bs], dev)
    t = time.time()
    for _ in range(3):
        p = forward_mlm(m, src, vals, pad)
    if dev.type == "mps":
        torch.mps.synchronize()
    print("plain bs", bs, "seq", src.shape[1], "s/forward", (time.time() - t) / 3, "out", tuple(p.shape), float(p[0, 1:6].mean()))
enable_attention_capture(m)
acc = []
set_capture(m, lambda w: acc.append(w.mean(1).sum(0).cpu()))
src, vals, pad = tok.batch(xb[:8], dev)
t = time.time()
p = forward_mlm(m, src, vals, pad)
if dev.type == "mps":
    torch.mps.synchronize()
print("attn bs 8 s/forward", time.time() - t, "captured", len(acc), tuple(acc[0].shape), "rowsum", float(acc[0][3].sum()) / 8)
vals2 = vals.clone()
vals2[:, 1:] = -1.0
set_capture(m, None)
p2 = forward_mlm(m, src, vals2, pad)
print("pred all-masked mean", float(p2[:, 1:].mean()), "pred visible mean", float(p[:, 1:].mean()), "true mean", float(vals[:, 1:].mean()))
mask = vals.clone()
idx = torch.randperm(vals.shape[1] - 1)[:400] + 1
mask[:, idx] = -1.0
p3 = forward_mlm(m, src, mask, pad)
err = (p3[:, idx] - vals[:, idx]).abs().mean()
base = (vals[:, idx] - vals[:, idx].mean()).abs().mean()
print("mae masked25%", float(err), "mae const", float(base))
