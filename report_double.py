"""Подвійний критерій: метод має пройти ОБИДВА зрівнювання, і кожне число
читається відносно ВЛАСНОЇ підлоги того самого режиму."""
import json, pathlib, sys
import numpy as np

REG = ["linear", "linear_hidden", "linear_rho0.5", "tanh", "relu"]
MARGIN = 0.03


def load(stat, reg):
    f = pathlib.Path(f"out/grid_{stat}_{reg}.json")
    fl = pathlib.Path(f"out/floor_grid_{stat}_{reg}.json")
    if not f.exists() or not fl.exists():
        return None, None
    return json.loads(f.read_text()), json.loads(fl.read_text())


regs = [r for r in REG if load("corr", r)[0] and load("mi", r)[0]]
g0, _ = load("corr", regs[0])
methods, structures = list(g0[list(g0)[0]]["methods"]), list(g0)

print(f"режими: {', '.join(regs)}\n")
verdict = {}
for st in structures:
    print(f"### {st}   (потрібно {g0[st]['correct']}, {g0[st]['pos']} проти {g0[st]['neg']})")
    print(f"{'режим':16}{'зрівн.':>8}{'підлога':>9}" + "".join(f"{m[:11]:>13}" for m in methods))
    ok = {m: 0 for m in methods}
    tot = 0
    for reg in regs:
        for stat in ("corr", "mi"):
            g, fl = load(stat, reg)
            f = fl[st]["floor"]
            row = f"{reg:16}{stat:>8}{f:>9.3f}"
            tot += 1
            # у режимі hidden задача НЕРОЗВʼЯЗНА за побудовою для трьох
            # структур: обумовлюватись нема на що. Правильна поведінка там —
            # сидіти на підлозі, а не перевершити її.
            unsolvable = reg.endswith("hidden") and st in ("confounder", "chain",
                                                           "two_hidden")
            for m in methods:
                v = g[st]["methods"][m]["main"]
                if g[st]["correct"] == "0.5":
                    good = abs(v - 0.5) < 0.06
                elif unsolvable:
                    good = abs(v - f) < 0.06
                else:
                    good = v > f + MARGIN
                ok[m] += good
                row += f"{v:>12.3f}{'*' if good else ' '}"
            print(row)
    passed = [m for m in methods if ok[m] == tot]
    verdict[st] = {"passed": passed, "score": {m: f"{ok[m]}/{tot}" for m in methods}}
    print(f"  пройшли ВСІ {tot} комбінацій: {', '.join(passed) if passed else 'НІХТО'}")
    print(f"  бали: " + "  ".join(f"{m[:9]} {ok[m]}/{tot}" for m in methods) + "\n")

print("### підсумок")
print(f"{'структура':13}" + "".join(f"{m[:11]:>13}" for m in methods))
tot_m = {m: 0 for m in methods}
for st in structures:
    print(f"{st:13}" + "".join(f"{verdict[st]['score'][m]:>13}" for m in methods))
    for m in methods:
        tot_m[m] += int(verdict[st]["score"][m].split("/")[0])
den = sum(int(verdict[structures[0]]["score"][methods[0]].split("/")[1])
          for _ in structures)
print(f"{'РАЗОМ':13}" + "".join(f"{str(tot_m[m])+'/'+str(den):>13}" for m in methods))
print()

pathlib.Path("out/double_criterion.json").write_text(
    json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
print("out/double_criterion.json")
