python run_finetune.py \
    --save_folder ./save/air/finetune_multi_weaver \
    --pretrained_dir ./save/air/pretrain_multi_weaver \
    --model_config_path model_configs.yaml \
    --pretrained_model_path ckpts/model_best.pth \
    --finetune_config_path configs/air/finetune.yaml \
    --evaluate_config_path configs/air/evaluate.yaml \
    --data_folder ./datasets/air \
    --n_runs 3 \
    --bootstrap_ratio 0.5 \
    --include_self 1 \
    --lr 0.0000001 \
    --epochs 50 \
    --only_evaluate True \