# python train_ms_tcn_looped_flow.py \
#     --data_dir ./data/2017-subaru-forester/preprocessed \
#     --save_folder ./save/ms_tcn_looped_flow \
#     --model_name ms_tcn_looped_flow \
#     --device cuda --gpu_id 0 \
#     --batch_size 128 --num_workers 4 --pin_memory \
#     --epochs 100 --learning_rate 1e-3 --weight_decay 1e-4 \
#     --amp --grad_clip 1.0 --print_freq 100 \
#     --scheduler cosine_restart --t0 10 --t_mult 2 \
#     --loss_name ce \
#     --train_steps 16 --act_weight 0.5 \
#     --pseudotarget_warmup 5000 \
#     --latent_noise_std 0.1 \
#     --noise_scale 1.0 \
#     --latent_dim 64 --denoiser_hidden_dim 128 \
#     --inference_steps 64 --inference_gamma 1.0 \
#     --inference_trajectories 1 \
#     --log_step_curve_freq 200

python train_looped_flow.py \
    --data_dir ./data/2017-subaru-forester/preprocessed \
    --save_folder ./save/progressive_reveal_v2 \
    --device cuda --gpu_id 0 \
    --batch_size 128 --num_workers 4 --pin_memory \
    --amp \
    --epochs 100 --lr 1e-3 --grad_clip 1.0 \
    --scheduler cosine_restart --t0 10 --t_mult 2 \
    --train_steps 8 --inference_steps 16 \
    --pseudotarget_warmup 5000 --latent_noise_std 0.1 \
    --noise_scale 1.0 \
    --num_reveal_groups 8 \
    --print_freq 100 --log_curve_freq 100