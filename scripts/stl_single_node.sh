#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@120
#SBATCH --exclude=a100-st-p4d24xlarge-2

DATASET_PATH="/data/home/lyuchen/swav_exp/new_stl10"
EXPERIMENT_PATH="./experiments/stl/deepcluster_knn0"
#EXPERIMENT_PATH="./experiments/stl/debug"
mkdir -p $EXPERIMENT_PATH

#srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label \
python -u main_deepclusterv2.py \
--nb_neighbor 0 \
--data_path $DATASET_PATH \
--nmb_crops 2 \
--size_crops 96 \
--min_scale_crops 0.33 \
--max_scale_crops 1. \
--crops_for_assign 0 1 \
--temperature 0.1 \
--hidden_mlp 1024 \
--feat_dim 128 \
--nmb_prototypes 512 \
--epochs 400 \
--batch_size 64 \
--base_lr 4.8 \
--final_lr 0.0048 \
--freeze_prototypes_niters 40000 \
--wd 0.000001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--arch resnet50 \
--sync_bn pytorch \
--dump_path $EXPERIMENT_PATH
