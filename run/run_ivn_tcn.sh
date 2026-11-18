#!/usr/bin/env bash
# IVN-TCN ablation (dims 64/128). Each run adds one component.
#
#   nohup bash run_ivn_tcn.sh > run_ivn_tcn.log 2>&1 &
#   tail -f run_ivn_tcn.log
#
GPU=0
DATA=./data/2017-subaru-forester/preprocessed
OUT=./save/ivn_tcn/forester
SEEDS="42"
mkdir -p "$OUT/logs"
done_already() { [ -f "$OUT/$1/test_at_best_val_report.txt" ] && echo "[skip] $1"; }
stamp() { date +%Y%m%d_%H%M%S; }

COMMON="--data_dir $DATA --gpu_id $GPU --epochs 100 --patience 15 \
    --dims 64 128 --num_blocks 1 1 --large_sizes 13 13 --small_size 5 \
    --ffn_ratio 2 --dropout 0.1 --lr 0.001 --batch_size 256 \
    --cpu_threads 4 --fast_data auto --amp bf16"

run() {
    local NAME=$1; shift
    done_already $NAME && return
    echo "[run ] $NAME  $(date '+%T')"
    python train_ivn_tcn.py $COMMON --save_folder $OUT/$NAME --seed $SEED "$@" \
        > $OUT/logs/${NAME}_$(stamp).log 2>&1
    echo "[done] $NAME  exit=$?  $(date '+%T')"
}

echo "Start $(date '+%F %T') on GPU $GPU"
for SEED in $SEEDS; do
    # A: light baseline (raw scalars, single-scale, plain head k=1)
    run A_base_s${SEED}          --no_embedding --no_multi_scale --ens_k 1
    # B: + RealMLP-style embedding (robust scale, smooth clip, PBLD, arb_id categorical)
    run B_emb_s${SEED}           --no_multi_scale --ens_k 1
    # C: + multi-scale temporal conv (kernels 3,5,13, per-channel fusion)
    run C_emb_ms_s${SEED}        --ms_kernel_sizes 3 5 13 --ms_fusion channel --ens_k 1
    # D: + TabM-style ensemble head (full model)
    run D_full_s${SEED}          --ms_kernel_sizes 3 5 13 --ms_fusion channel --ens_k 8
    # E: full model but arb_id treated as numeric (tests categorical ID embedding)
    run E_full_numID_s${SEED}    --ms_kernel_sizes 3 5 13 --ms_fusion channel --ens_k 8 --cat_col -1
done
echo "All runs finished $(date '+%F %T')"