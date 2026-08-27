import argparse
import sys
import pathlib
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо
from eval.sweep import sweep, table

ap = argparse.ArgumentParser()
ap.add_argument("--levels", default="9.0,8.0,7.0,6.14,5.0,4.0,3.0")
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--trees", action="store_true")
ap.add_argument("--tag", default="sparsity")
ap.add_argument("--ideal", action="store_true", help="додати ідеальний рівень (0% нулів)")
a = ap.parse_args()

levels = [float(x) for x in a.levels.split(",")]
if a.ideal:
    levels = [None] + levels
r = sweep(levels, a.seeds, a.trees, a.tag)
print(table(r))
print(f"\nout/sweep_{a.tag}.json")
