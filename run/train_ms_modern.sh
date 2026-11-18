#!/usr/bin/env bash
# ModernTCN v2: borrow two design choices from MS-TCN-RetNet (no RetNet):
#   (1) multi-scale fusion by concat + learnable projection instead of sum
#   (2) temporal mean pooling head instead of flatten
# Ablation isolates each change against vanilla (F1=0.974, flatten head).
#
#   nohup bash run_ms_v2.sh > run_ms_v2.log 2>&1 &
#   tail -f run_ms_v2.log
#
GPU=0
DATA=./data/2017-subaru-forester/preprocessed
OUT=./save/ms_moderntcn/forester
SEEDS="42"
mkdir -p "$OUT/logs"
done_already() { [ -f "$OUT/$1/test_at_best_val_report.txt" ] && echo "[skip] $1"; }
stamp() { date +%Y%m%d_%H%M%S; }

COMMON="--data_dir $DATA --gpu_id $GPU \
    --epochs 100 --patience 15 \
    --dims 128 256 --num_blocks 1 1 \
    --large_sizes 13 13 --small_size 5 \
    --ffn_ratio 2 --dropout 0.1 --class_dropout 0.0 \
    --lr 0.001 --batch_size 256 --num_workers 4 --pin_memory"

run() {
    local NAME=$1; shift
    done_already $NAME && return
    echo "[run ] $NAME  $(date '+%T')"
    python train_ms_modern_tcn.py $COMMON --save_folder $OUT/$NAME --seed $SEED "$@" \
        > $OUT/logs/${NAME}_$(stamp).log 2>&1
    echo "[done] $NAME  exit=$?  $(date '+%T')"
}

echo "Start $(date '+%F %T') on GPU $GPU"

for SEED in $SEEDS; do
    # # A: head effect alone (single-scale + mean head)
    # run vanilla_mean_s${SEED}         --no_multi_scale --head_type mean

    # # B: fusion effect alone (concat multi-scale + original flatten head)
    # run ms_cat_3_5_7_9_13_flat_s${SEED} --multi_scale --ms_kernel_sizes 3 5 7 9 13 \
    #                                     --ms_fusion concat --head_type flatten

    # C: both changes (main candidate)
    run ms_cat_3_5_7_9_13_mean_s${SEED} --multi_scale --ms_kernel_sizes 3 5 7 9 13 \
                                        --ms_fusion concat --head_type mean

    # D: lighter kernel set, both changes
    run ms_cat_3_5_13_mean_s${SEED}     --multi_scale --ms_kernel_sizes 3 5 13 \
                                        --ms_fusion concat --head_type mean
done

echo "All runs finished $(date '+%F %T')"