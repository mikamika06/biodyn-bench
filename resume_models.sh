#!/bin/bash
# Довчити модельну частину до 6 зерен. Відновлюваний:
# усе готове пропускається, перезапуск продовжує з місця обриву.
cd "$(dirname "$0")"
PANELS="confounder confounder_hidden collider chain chain_hidden"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
LOG=out/resume.log
mkdir -p out

todo=0; done_=0
for s in $SEEDS; do for p in $PANELS; do
  tag=$p; [ "$s" != "0" ] && tag="${p}_s${s}"
  if [ -f "out/model_columns_${tag}.json" ]; then done_=$((done_+1)); else todo=$((todo+1)); fi
done; done
echo "[$(date '+%H:%M')] готово $done_, лишилось $todo" | tee -a $LOG

for s in $SEEDS; do
  for p in $PANELS; do
    tag=$p; [ "$s" != "0" ] && tag="${p}_s${s}"
    if [ -f "out/model_columns_${tag}.json" ]; then continue; fi

    if [ ! -f "out/model_${tag}.pt" ]; then
      echo "[$(date '+%H:%M')] навчання $tag" | tee -a $LOG
      python train_model.py --panel "$p" --seed "$s" --genes small --steps 8000 \
        --layers 2 --d 128 --lr 1e-3 --out "model_${tag}.pt" \
        > "out/train_${tag}.log" 2>&1 || { echo "  ЗБІЙ навчання $tag" | tee -a $LOG; continue; }
    fi

    echo "[$(date '+%H:%M')] оцінка $tag" | tee -a $LOG
    python eval_model.py --ckpt "model_${tag}.pt" --out "model_columns_${tag}.json" \
      > "out/eval_${tag}.log" 2>&1 || { echo "  ЗБІЙ оцінки $tag" | tee -a $LOG; continue; }
    ./notify.sh "$tag готово"
  done
done
echo "[$(date '+%H:%M')] усе завершено" | tee -a $LOG
python report_model.py | tee out/report_model.txt
