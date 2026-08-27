import json, pathlib
import numpy as np

rows = []
for f in sorted(pathlib.Path("out").glob("model_columns_sp*.json")):
    lib = f.stem.replace("model_columns_sp", "")
    r = json.loads(f.read_text())["confounder"]
    m = r["model_truth"]["means"]
    denom = m["D"] - m["N"]
    ratio = (m["C"] - m["N"]) / denom if denom > 0.05 else float("nan")
    rows.append((float(lib), r, ratio))
rows.sort(key=lambda t: -t[0])

ideal = pathlib.Path("out/model_columns_confounder.json")
if ideal.exists():
    r = json.loads(ideal.read_text())["confounder"]
    m = r["model_truth"]["means"]
    rows.insert(0, (float("inf"), r, (m["C"] - m["N"]) / (m["D"] - m["N"])))

print(f"{'lib':>6}{'відношення':>12}{'ІСТИНА':>9}{'увага':>9}{'градієнт':>10}{'проба':>8}"
      f"{'вірн.гр':>9}{'контроль':>10}")
print("-" * 73)
for lib, r, ratio in rows:
    lab = "ідеал" if lib == float("inf") else f"{lib:.2f}"
    mt = r["model_truth"]
    g = lambda nm, k="vs_planted": r["methods"][nm][k] if nm in r["methods"] else float("nan")
    print(f"{lab:>6}{ratio:>12.3f}{mt['vs_planted']:>9.3f}{g('увага'):>9.3f}"
          f"{g('градієнт'):>10.3f}{g('проба'):>8.3f}"
          f"{g('градієнт','spearman_vs_model_truth'):>9.3f}{mt['control']:>10.3f}")
print("\nвідношення: 0 = модель ігнорує хибний звʼязок, 1 = користується нарівні")
print("вірн.гр = рангова вірність градієнта")
