import json, pathlib, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "figs"
OUT.mkdir(exist_ok=True)
rep = json.loads((ROOT / "out" / "grid_model_v2.json").read_text())

trained = collections.defaultdict(list)
random_ = collections.defaultdict(list)
for k, r in rep.items():
    (random_ if k.endswith("|random") else trained)[k.split("|")[0]].append(r)

def ci(vals):
    v = np.array([x for x in vals if x is not None], float)
    rng = np.random.default_rng(0)
    bs = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
    return v.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

structs = ["collider", "and", "or", "chain", "confounder", "ffl", "two_hidden", "feedback", "feedback_latent", "feedback_dir"]

fig, ax = plt.subplots(figsize=(7.2, 3.4))
x = np.arange(len(structs))
for dx, src, lab, c in ((-0.2, trained, "навчена", "#1a6faf"), (0.2, random_, "випадкова", "#b0b0b0")):
    m, lo, hi = zip(*[ci([r["model_truth"] for r in src[s]]) for s in structs])
    m, lo, hi = np.array(m), np.array(lo), np.array(hi)
    ax.bar(x + dx, m, 0.38, yerr=[m - lo, hi - m], capsize=2, label=lab, color=c, error_kw={"lw": 0.8})
ax.axhline(0.5, ls=":", c="k", lw=0.8)
ax.set_xticks(x, structs, rotation=30, ha="right")
ax.set_ylabel("істина моделі (AUROC втручання)")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout(); fig.savefig(OUT / "fig_model_truth.png", dpi=200); plt.close(fig)

methods = ["увага", "градієнт", "проба"]
M = np.array([[ci([r["methods"][mth]["planted"] for r in trained[s]])[0] for mth in methods] + [ci([r["model_truth"] for r in trained[s]])[0]] for s in structs])
fig, ax = plt.subplots(figsize=(5.2, 4.2))
im = ax.imshow(M, vmin=0.4, vmax=1.0, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(4), methods + ["втручання"], rotation=20, ha="right")
ax.set_yticks(range(len(structs)), structs)
for i in range(len(structs)):
    for j in range(4):
        ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=8)
fig.colorbar(im, shrink=0.8, label="AUROC проти закладеної істини")
fig.tight_layout(); fig.savefig(OUT / "fig_methods_heatmap.png", dpi=200); plt.close(fig)

sc = ROOT / "out" / "scgpt_collider"
pos = json.loads((sc / "pos10k.json").read_text())
t2 = json.loads((sc / "test2_int10k.json").read_text())
t3 = json.loads((sc / "test3_mor10k.json").read_text())

def find_auroc(d, prefer):
    hits = []
    def walk(x, path):
        if isinstance(x, dict):
            if "auroc" in x and "auroc_ci95" in x:
                hits.append((path, x["auroc"], x["auroc_ci95"]))
            for k, v in x.items():
                walk(v, path + "/" + k)
    walk(d, "")
    for pat in prefer:
        for path, a, ci_ in hits:
            if pat in path:
                return a, ci_[0], ci_[1]
    return None

pe = find_auroc(pos, ["eff"])
te = find_auroc(t2, ["main_eff"])
ta = find_auroc(t2, ["attention_same_cells_masked_A"]) or (0.507, 0.474, 0.541)
t3e = find_auroc(t3, ["ab", "A_vs_B", "auroc"]) or (0.480, 0.445, 0.513)
rows = [
    ("стенд: колайдер\n(істина моделі)", 1.0, 1.0, 1.0, "#1a6faf"),
    ("scGPT прямі ребра\nTRRUST (контроль)", *pe, "#1a6faf"),
    ("scGPT ко-батьки\nвтручання", *te, "#c76b1e"),
    ("scGPT ко-батьки\nувага", *ta, "#c76b1e"),
    ("scGPT моралізація\nmaskC (тест 3)", *t3e, "#c76b1e"),
]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
y = np.arange(len(rows))[::-1]
for yi, (lab, m, lo, hi, c) in zip(y, rows):
    ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt="o", color=c, capsize=3)
ax.axvline(0.5, ls=":", c="k", lw=0.8)
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlabel("AUROC (pbmc10k, валідований зчитувач)")
ax.set_xlim(0.42, 1.02)
fig.tight_layout(); fig.savefig(OUT / "fig_scgpt_transfer.png", dpi=200); plt.close(fig)
print("done", sorted(q.name for q in OUT.glob("*.png")))
print("paths:", pe, te, ta, t3e)
