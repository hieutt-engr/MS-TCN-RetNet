#!/usr/bin/env bash
# ID-conditioned embedding ablation, on top of run C (embedding + multi-scale, plain head).
#   C_fix : run C with the fixed arb_id vocabulary (normal traffic, frame-fraction threshold)
#   F     : C_fix + ID-conditioned (FiLM) modulation of the numeric field embeddings
# C_fix is needed because the old C used the flawed vocabulary (capped at 512 IDs);
# F vs C_fix then isolates the effect of ID conditioning alone.
#
#   nohup bash run_idcond.sh > run_idcond.log 2>&1 &
#   tail -f run_idcond.log
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
    --cpu_threads 4 --fast_data auto --amp bf16 \
    --ms_kernel_sizes 3 5 13 --ms_fusion channel --ens_k 1 \
    --vocab_from normal --vocab_min_frac 1e-5"

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
    run C_fix_s${SEED}
    run F_idcond_s${SEED}   --id_cond
done
echo "All runs finished $(date '+%F %T')"
