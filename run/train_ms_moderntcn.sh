#!/usr/bin/env bash
# Multi-Scale ModernTCN for CAN bus IDS (multi-class).
# Runs multi-scale variant then vanilla baseline for direct comparison.
#
#   nohup bash run_ms_moderntcn.sh > run_ms_moderntcn.log 2>&1 &
#   tail -f run_ms_moderntcn.log
#   tail -f save/moderntcn/forester/logs/*.log
#
GPU=0
DATA=./data/2017-subaru-forester/preprocessed
OUT=./save/moderntcn/forester
SEEDS="42"
mkdir -p "$OUT/logs"
done_already() { [ -f "$OUT/$1/test_at_best_val_report.txt" ] && echo "[skip] $1"; }
stamp() { date +%Y%m%d_%H%M%S; }

echo "Start $(date '+%F %T') on GPU $GPU"

# ── Run 1: Multi-Scale DWConv (kernels 3, 7, 13) ──
for SEED in $SEEDS; do
    NAME=ms_k3_7_13_dim128_s${SEED}
    done_already $NAME && continue
    echo "[run ] $NAME  $(date '+%T')"
    python train_ms_modern_tcn.py \
        --data_dir $DATA \
        --save_folder $OUT/$NAME \
        --gpu $GPU --cpu_threads 4 \
        --epochs 100 --patience 15 \
        --dims 128 256 --num_blocks 1 1 \
        --large_sizes 13 13 --small_size 5 \
        --multi_scale --ms_kernel_sizes 3 7 13 \
        --ffn_ratio 2 --dropout 0.1 --class_dropout 0.0 \
        --lr 0.001 --batch_size 256 \
        --seed $SEED \
        > $OUT/logs/${NAME}_$(stamp).log 2>&1
    echo "[done] $NAME  exit=$?  $(date '+%T')"
done

# ── Run 2: Multi-Scale DWConv (kernels 3, 5, 9, 13) ──
# for SEED in $SEEDS; do
#     NAME=ms_k3_5_9_13_dim128_s${SEED}
#     done_already $NAME && continue
#     echo "[run ] $NAME  $(date '+%T')"
#     python train_ms_modern_tcn.py \
#         --data_dir $DATA \
#         --save_folder $OUT/$NAME \
#         --gpu $GPU --cpu_threads 4 \
#         --epochs 100 --patience 15 \
#         --dims 128 256 --num_blocks 1 1 \
#         --large_sizes 13 13 --small_size 5 \
#         --multi_scale --ms_kernel_sizes 3 5 9 13 \
#         --ffn_ratio 2 --dropout 0.1 --class_dropout 0.0 \
#         --lr 0.001 --batch_size 256 \
#         --seed $SEED \
#         > $OUT/logs/${NAME}_$(stamp).log 2>&1
#     echo "[done] $NAME  exit=$?  $(date '+%T')"
# done

# # ── Run 3: Vanilla baseline (single-scale, for comparison) ──
# for SEED in $SEEDS; do
#     NAME=vanilla_dim128_s${SEED}
#     done_already $NAME && continue
#     echo "[run ] $NAME  $(date '+%T')"
#     python train_ms_modern_tcn.py \
#         --data_dir $DATA \
#         --save_folder $OUT/$NAME \
#         --gpu $GPU --cpu_threads 4 \
#         --epochs 100 --patience 15 \
#         --dims 128 256 --num_blocks 1 1 \
#         --large_sizes 13 13 --small_size 5 \
#         --no_multi_scale \
#         --ffn_ratio 2 --dropout 0.1 --class_dropout 0.0 \
#         --lr 0.001 --batch_size 256 \
#         --seed $SEED \
#         > $OUT/logs/${NAME}_$(stamp).log 2>&1
#     echo "[done] $NAME  exit=$?  $(date '+%T')"
# done

echo "All runs finished $(date '+%F %T')"