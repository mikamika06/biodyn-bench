import json, pathlib, sys
import numpy as np

OUT = pathlib.Path("out")
LABEL = {"confounder": "1 конфаундер, A видно", "confounder_hidden": "2 конфаундер, A приховано",
         "collider": "3 колайдер", "chain": "4a ланцюг, B видно",
         "chain_hidden": "4b ланцюг, B приховано",
         "confounder_full": "1* те саме, 155 генів"}
PAIR = {"confounder": ("D", "C"), "confounder_hidden": ("D", "C"),
        "collider": ("K", "N"), "chain": ("D", "T"), "chain_hidden": ("D", "T"),
        "confounder_full": ("D", "C")}
ORDER = ["confounder", "confounder_full", "confounder_hidden", "collider", "chain", "chain_hidden"]

rows = {}
for f in OUT.glob("model_columns_*.json"):
    for panel, res in json.loads(f.read_text()).items():
        rows.setdefault(panel, []).append(res)

if not rows:
    sys.exit("немає результатів")

print(f"{'структура':26}{'увага':>10}{'градієнт':>11}{'проба':>9}{'ІСТИНА':>9}{'n':>4}")
print("-" * 69)
for p in ORDER:
    if p not in rows:
        continue
    rs = rows[p]
    def mm(nm):
        v = [r["methods"][nm]["vs_planted"] for r in rs if nm in r["methods"]]
        return np.mean(v) if v else float("nan")
    t = np.mean([r["model_truth"]["vs_planted"] for r in rs])
    print(f"{LABEL[p]:26}{mm('увага'):>10.3f}{mm('градієнт'):>11.3f}"
          f"{mm('проба'):>9.3f}{t:>9.3f}{len(rs):>4}")

print(f"\n{'структура':26}{'ефект D/K':>11}{'ефект C/T':>11}{'ефект N':>10}{'відношення':>12}")
print("-" * 70)
for p in ORDER:
    if p not in rows:
        continue
    m = rows[p][0]["model_truth"]["means"]
    pk, nk = PAIR[p]
    pos, neg, nul = m.get(pk), m.get(nk), m.get("N", float("nan"))
    if p == "collider":
        pos, neg = m["D"], m["K"]
    print(f"{LABEL[p]:26}{pos:>11.4f}{neg:>11.4f}{nul:>10.4f}"
          f"{(neg-nul)/max(pos-nul,1e-9):>12.3f}")
print("\nвідношення = (ефект несправжньої пари − шум) / (ефект справжньої − шум)")
print("0 = модель ігнорує несправжній звʼязок;  1 = користується ним нарівні")
