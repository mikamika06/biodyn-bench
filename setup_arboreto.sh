#!/bin/zsh
cd /Users/macbook/biodyn-bench
python -m venv .venv_arb
.venv_arb/bin/pip install -q --upgrade pip
.venv_arb/bin/pip install -q "numpy<2" "pandas<2" "scikit-learn<1.4" \
  "dask==2022.12.1" "distributed==2022.12.1" "arboreto==0.1.6" 2>&1 | tail -5
.venv_arb/bin/python -c "import arboreto, dask, distributed, sklearn, numpy, pandas; print('arb env ok  dask', dask.__version__, 'sklearn', sklearn.__version__)"
