# biodyn-bench

Synthetic benchmark for causal structure recovery in single-cell gene expression,
and for interpretability methods applied to models trained on it.

The point is a **positive control**: data where the regulatory structure is known
because we planted it, so that a pipeline can be asked whether it recovers
what is certainly there.

## What makes it different from existing simulators

Simulators with known networks already exist (SERGIO, BEELINE/BoolODE, dyngen,
GeneNetWeaver). They benchmark *network inference algorithms* against an edge list.
Ground-truth-circuit benchmarks also exist in mechanistic interpretability
(Tracr, InterpBench, MIB). They contain no biology.

This stand sits at the intersection and adds four things:

**Matched pairs.** Edge and non-edge pairs are tuned to equal association
strength, so a method cannot win by reading magnitude. Without this, a benchmark
partly measures expression level rather than regulation.

**Measured floor.** Every column reports what a method that understands nothing
would score. Across 64 configurations the floor predicts Pearson correlation to a
median of 0.0035 (r = 0.943). Numbers are read against the floor, not against 0.5.

**Non-edge columns.** Curated databases list edges, never non-edges, so
"did the method invent an edge?" cannot be asked on real data. Here the collider
column has the correct answer 0.5 by construction.

**Two truths per cell.** Both what was planted in the data and what the trained
model actually uses (measured by input intervention), so "the method is broken"
can be separated from "the model learned something false and the method
reported it honestly".

## Layout

```
src/sim/       generators
  grid.py        nine causal structures, matched pairs, reference non-edges
  network.py     connected scale-free DAG, cell-type mixtures
  hill.py        Hill-kinetics generator (independent replication)
  realistic.py   Splatter-style count hierarchy
  counts.py      latent values to counts to normalised expression
src/methods/   correlation, MI, partial correlation, ARACNe, trees,
               graphical lasso, pairwise LiNGAM
src/model/     scGPT-style masked transformer + interpretability hooks
src/eval/      AUROC, post-hoc strength matching

runs/          experiments
  grid_run.py       classical methods across structures
  floor_grid.py     floor per structure and regime
  grid_model.py     two truths per structure
  net_run.py        connected graph, p/n sweep
  net_model.py      two truths inside a connected graph
  mix_run.py        cell-type mixtures
  depth_run.py      read depth and gene-level heterogeneity
  ablate.py         exhaustive head/neuron ablation, pairwise redundancy
  identify.py       identifiability by matching impossibility
  identify_auroc.py the same, restated as an AUROC bound
  ident_spread.py   per-seed spread of the residual
  regime.py         regime calculator for a real count matrix

reports/       consolidation
  report_all.py     everything into out/RESULTS.md
  report_double.py  dual matching criterion against per-regime floors
  report_noise.py   signed method strength across noise shapes
  compare_ident.py  identifiability against measured method power

validation/    reference checks
  validate_trees.py, ref_trees.py, compare_trees.py   our trees vs arboreto

legacy/        superseded by the grid; kept because old results reference it
run_all.sh     sequential work queue, every stage resumable
```

Scripts change directory to the repository root on import, so `out/` and
`data/` resolve the same way from any working directory.

## Quick start

```bash
python runs/grid_run.py --seeds 20         # classical methods, ~90 s
python runs/floor_grid.py --seeds 20       # floors for the same run
python reports/report_double.py            # dual matching criterion
python runs/net_run.py --seeds 4           # connected graph, p/n sweep
python runs/identify.py --seeds 3          # is the answer in the data at all
python runs/regime.py --counts your.npy    # what regime is a real dataset in
./run_all.sh                               # the full queue
```

No data files are stored. Everything is generated from a seed, so a result is
reproducible from forty characters of configuration.

## Selected measurements

```
conditioning methods degrade monotonically with p/n         0.85 → 0.58
the collider artifact grows with sample size                0.54 → 0.89
graphical lasso does not rescue the p/n regime              0.58 at p/n = 1
the collider artifact is 5x weaker under saturating kinetics
gene-level expression heterogeneity, not sparsity, is what
  destroys causal discrimination                            0.84 → 0.54 at equal zeros
edge direction is identifiable iff latent noise is non-Gaussian;
  the scRNA count pipeline neither creates nor destroys it
on pbmc3k no gene panel has causal headroom above +0.09,
  and the transcription-factor panel has +0.027
```

Details and caveats live in the working notes, which are not published in this
repository.
