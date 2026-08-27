"""Сила методу = |AUROC − підлога|. Симетрична міра: перевернутий метод
інформативний так само, як прямий, треба лише знати знак."""
import json, pathlib
import numpy as np

NOISES = ["gauss", "uniform", "laplace", "exp"]
STATS = ["corr", "mi"]
STRUCT = ["confounder", "chain", "collider", "feedback_dir"]


def load(nd, stat):
    g = pathlib.Path(f"out/grid_{nd}_{stat}_linear.json")
    f = pathlib.Path(f"out/floor_grid_{nd}_{stat}_linear.json")
    if not (g.exists() and f.exists()):
        return None, None
    return json.loads(g.read_text()), json.loads(f.read_text())


g0 = next((load(n, s)[0] for n in NOISES for s in STATS if load(n, s)[0]), None)
if g0 is None:
    raise SystemExit("немає даних")
methods = list(g0[list(g0)[0]]["methods"])

print("сила = |AUROC − підлога|;  (−) означає перевернутий знак\n")
for st in STRUCT:
    print(f"### {st}")
    print(f"{'шум':>9}{'зрівн':>6}{'підлога':>9}" + "".join(f"{m[:11]:>13}" for m in methods))
    agg = {m: [] for m in methods}
    for nd in NOISES:
        for stat in STATS:
            g, f = load(nd, stat)
            if not g or st not in g:
                continue
            fl = f[st]["floor"]
            row = f"{nd:>9}{stat:>6}{fl:>9.3f}"
            for m in methods:
                v = g[st]["methods"][m]["main"]
                pw = abs(v - fl)
                agg[m].append(pw)
                row += f"{('-' if v < fl else '') + f'{pw:.3f}':>13}"
            print(row)
    print(f"{'СЕРЕДНЄ':>15}{'':>9}" + "".join(
        f"{np.mean(agg[m]) if agg[m] else float('nan'):>13.3f}" for m in methods))
    print()

# Для стовпця колайдера велика сила = ПРОВАЛ (метод вигадав ребро).
# Для решти велика сила = робота. Тому агрегувати треба зі знаком.
print("### підсумок зі знаком")
print("   сила там, де треба розрізнити, МІНУС сила там, де треба мовчати\n")
gain = {m: [] for m in methods}
cost = {m: [] for m in methods}
for st in STRUCT:
    for nd in NOISES:
        for stat in STATS:
            g, f = load(nd, stat)
            if not g or st not in g:
                continue
            silent = g[st]["correct"] == "0.5"
            for m in methods:
                pw = abs(g[st]["methods"][m]["main"] - f[st]["floor"])
                (cost if silent else gain)[m].append(pw)
print(f"{'метод':17}{'розрізняє':>11}{'вигадує':>10}{'чистий':>9}")
rows = []
for m in methods:
    a = np.mean(gain[m]) if gain[m] else 0.0
    b = np.mean(cost[m]) if cost[m] else 0.0
    rows.append((a - b, m, a, b))
for net, m, a, b in sorted(rows, reverse=True):
    print(f"  {m:15}{a:>11.3f}{b:>10.3f}{net:>9.3f}")
