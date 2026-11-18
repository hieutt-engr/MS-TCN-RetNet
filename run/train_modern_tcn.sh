#!/usr/bin/env bash
# ModernTCN for CAN bus IDS (multi-class), speed-optimized training script.
# One GPU, runs one after another:
#
#   nohup bash run_moderntcn.sh > run_moderntcn.log 2>&1 &
#   tail -f run_moderntcn.log                 # overall progress
#   tail -f save/moderntcn/2017-subaru-forester/logs/*.log  # per-run log
#
# Data: .pt files (train/val/test) from preprocess_can_ml_subset.py.
# Runs that already have test_at_best_val_report.txt (written at the very end
# of training) are skipped, so a crashed or killed run is simply restarted.
GPU=0
DATA=./data/2017-subaru-forester/preprocessed
OUT=./save/moderntcn/forester
SEEDS="42"                                        
mkdir -p "$OUT/logs"
done_already() { [ -f "$OUT/$1/test_at_best_val_report.txt" ] && echo "[skip] $1"; }
stamp() { date +%Y%m%d_%H%M%S; }

echo "Start $(date '+%F %T') on GPU $GPU"

for SEED in $SEEDS; do
    NAME=mtcn_bs256_dim128_low_dr_lr0.001_s${SEED}
    done_already $NAME && continue
    echo "[run ] $NAME  $(date '+%T')"
    python train_moderntcn.py \
        --data_dir $DATA \
        --save_folder $OUT/$NAME \
        --gpu $GPU --cpu_threads 4 \
        --epochs 100 --patience 15 \
        --dims 128 256 --num_blocks 1 1 \
        --large_sizes 13 13 --small_size 5 \
        --ffn_ratio 2 --dropout 0.1 --class_dropout 0.0 \
        --lr 0.001 --batch_size 256 \
        --seed $SEED \
        > $OUT/logs/${NAME}_$(stamp).log 2>&1
    echo "[done] $NAME  exit=$?  $(date '+%T')"
done

echo "All runs finished $(date '+%F %T')"