export CUDA_VISIBLE_DEVICES=5

#!/bin/bash

tasks=('bace' 'bbbp' 'clintox' 'sider' 'tox21' 'toxcast')

for task in "${tasks[@]}"
do
    python main.py finetune \
        --data_path exampledata/finetune/${task}.csv \
        --features_path exampledata/finetune/${task}_kge.npy \
        --save_dir /data/pj20/grover/finetune_kge/1200_new/${task} \
        --checkpoint_path /data/pj20/grover/pretrain/grover_large.pt \
        --dataset_type classification \
        --split_type scaffold_balanced \
        --ensemble_size 1 \
        --num_folds 3 \
        --no_features_scaling \
        --ffn_hidden_size 200 \
        --batch_size 32 \
        --epochs 20 \
        --init_lr 0.00015
done


# for task in "${tasks[@]}"
# do
#     python main.py finetune \
#         --data_path exampledata/finetune/${task}.csv \
#         --features_path exampledata/finetune/${task}_kge_200.npy \
#         --save_dir /data/pj20/grover/finetune_kge/200/${task} \
#         --checkpoint_path /data/pj20/grover/pretrain/grover_large.pt \
#         --dataset_type classification \
#         --split_type scaffold_balanced \
#         --ensemble_size 1 \
#         --num_folds 3 \
#         --no_features_scaling \
#         --ffn_hidden_size 200 \
#         --batch_size 32 \
#         --epochs 20 \
#         --init_lr 0.00015
# done