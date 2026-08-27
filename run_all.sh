#!/bin/zsh
cd /Users/macbook/biodyn-bench
export PYTHONWARNINGS=ignore
log() { echo "[$(date +%H:%M)] $1" >> out/RUN.log; }

log "A: модель, 6 зерен, 9 структур"
python grid_model.py --seeds 6 --panel 10 --d 128 --layers 2 --steps 8000 \
  --out grid_model_s12.json >> out/model_s12.log 2>&1
log "A готово"

log "B: ідентифіковність — решта стовпців"
python identify.py --seeds 3 --stats corr,mi,sq,partial --noises gauss,laplace \
  --only collider,feedback_dir,and,or --out identify_rest.json >> out/ident_rest.log 2>&1
log "C: розкид по зернах"
python ident_spread.py --seeds 8 --only confounder,feedback_dir,and,chain \
  --noises gauss,laplace >> out/spread.log 2>&1
log "D: межа у шкалі AUROC"
python identify_auroc.py --seeds 3 --only confounder,chain,two_hidden,and,or,ffl,feedback_dir \
  --noises gauss,laplace >> out/auroc_bound.log 2>&1
python identify_auroc.py --seeds 3 --hide --only confounder,chain,two_hidden \
  --noises gauss,laplace >> out/auroc_bound.log 2>&1
log "D готово"

log "E: дві істини у ЗВʼЯЗНОМУ графі"
python net_model.py --seeds 3 --genes 150 --cells 4000 --steps 20000 \
  --out net_model.json >> out/net_model.log 2>&1
log "E готово"

log "F: вичерпна абляція, задача 106"
for st in confounder chain collider two_hidden ffl and or feedback_dir; do
  python ablate.py --structure $st --steps 8000 --out ablation.json >> out/ablation.log 2>&1
  log "  абляція $st"
done
log "F готово"

log "G: прохід 2 — структури з еталонними парами, більша місткість"
python grid_model.py --seeds 6 --panel 10 --d 192 --layers 4 --steps 20000 \
  --only ffl,feedback,or,feedback_dir --out grid_model_pass2.json >> out/model_pass2.log 2>&1
log "G готово"

log "H: модель у режимах tanh і rho"
python grid_model.py --seeds 3 --panel 10 --link tanh --out grid_model_s12.json >> out/model_s12.log 2>&1
python grid_model.py --seeds 3 --panel 10 --rho 0.5 --out grid_model_s12.json >> out/model_s12.log 2>&1
log "ВСЕ ЗАВЕРШЕНО"
