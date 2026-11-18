# python train_ms_tcn_retnet_free_at.py \
#   --data_dir ./data/2017-subaru-forester/preprocessed \
#   --save_folder ./save/ms_tcn_retnet_free_at \
#   --model_name ms_tcn_retnet_free_at \
#   --device cuda \
#   --gpu_id 1 \
#   --epochs 100 \
#   --batch_size 128 \
#   --learning_rate 1e-4 \
#   --epsilon 4/255 \
#   --free_replays 4 \
#   --free_step_size 4/255 \
#   --adv_eval_freq 1 \
#   --attack_steps_eval 20 \
#   --amp \
#   --pin_memory \
#   --print_freq 100 \
#   --epoch_save_every 5 > ./save/log/train_fgsm_rs_at_free.log 2>&1 &


python train_ms_tcn_retnet_free_at_adv_ndcl.py \
  --data_dir ./data/2017-subaru-forester/preprocessed \
  --save_folder ./save/ms_tcn_retnet_free_at_adv_ndcl \
  --model_name ms_tcn_retnet_free_at_adv_ndcl \
  --device cuda \
  --gpu_id 0 \
  --epochs 100 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --epsilon 4/255 \
  --free_replays 4 \
  --free_step_size 4/255 \
  --adv_eval_freq 1 \
  --attack_steps_eval 20 \
  --amp \
  --pin_memory \
  --adv_ce_weight 1.0 \
  --kl_weight 0.1 \
  --feature_consistency_weight 0.0 \
  --adv_ndcl_weight 0.003 \
  --adv_ndcl_temperature 0.1 \
  --adv_ndcl_margin 0.2 \
  --adv_ndcl_hard_ratio 0.10 \
  --adv_ndcl_anchor_mode attack_only \
  --adv_ndcl_negative_mode normal_for_attack \
  --adv_ndcl_warmup_epochs 5 \
  --adv_ndcl_ramp_epochs 10 > ./save/log/train_ms_tcn_retnet_free_at_adv_ndcl.log 2>&1 &
