import argparse
import sys
import pathlib

import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # out/ і data/ рахуються від кореня репо

import bench
from eval.columns import set_link

parser = argparse.ArgumentParser(description="Прогін стенду")
parser.add_argument("--cheap", action="store_true", help="без деревних методів")
parser.add_argument("--seeds", type=int, default=None, help="кількість зерен")
parser.add_argument("--out", default=None)
parser.add_argument("--link", default="linear", choices=["linear","tanh","relu"])
parser.add_argument("--match", default="corr", choices=["corr","mi"])
args = parser.parse_args()

set_link(args.link, args.match)

if args.seeds:
    bench.CHEAP_SEEDS = bench.TREE_SEEDS = args.seeds
only = list(bench.CHEAP) if args.cheap else None

partial = args.cheap or args.seeds or args.link != "linear" or args.match != "corr"
out = args.out or (f"bench_{args.link}_{args.match}.json" if partial else "bench_classical.json")
if partial and not args.out:
    print(f"неповний прогін -> out/{out} (повний файл не чіпаю)\n")

report = bench.run(only=only)
print(bench.table(report))
print("\n" + str(bench.save(report, out)))
