#!/bin/bash
#SBATCH --partition=a100
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --job-name=deepclusterv2_400ep_2x224_pretrain
#SBATCH --time=48:00:00
#SBATCH --exclude=a100-st-p4d24xlarge-35

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

DATASET_PATH="/datasets01/imagenet_full_size/061417"
EXPERIMENT_PATH="./experiments/imagenet/deepclusterv2"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label \
python -u main_deepclusterv2_imagenet.py \
--knn_epoch 0 \
--nb_neighbor 0 \
--data_path $DATASET_PATH \
--nmb_crops 2 \
--size_crops 224 \
--min_scale_crops 0.08 \
--max_scale_crops 1. \
--crops_for_assign 0 1 \
--temperature 0.1 \
--feat_dim 128 \
--nmb_prototypes 3000 3000 3000 \
--epochs 400 \
--batch_size 64 \
--base_lr 1.2 \
--final_lr 0.0012 \
--freeze_prototypes_niters 300000 \
--wd 0.000001 \
--workers 12 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--dist_url $dist_url \
--arch resnet50 \
--sync_bn apex \
--dump_path $EXPERIMENT_PATH

#EXPERIMENT_PATH="./experiments/debug"
#mkdir -p $EXPERIMENT_PATH
#python -m torch.distributed.launch --nproc_per_node=1 \
#main_deepclusterv2_imagenet.py \
#--knn_epoch 0 \
#--nb_neighbor 0 \
#--data_path $DATASET_PATH \
#--nmb_crops 2 \
#--size_crops 224 \
#--min_scale_crops 0.08 \
#--max_scale_crops 1. \
#--crops_for_assign 0 1 \
#--temperature 0.1 \
#--feat_dim 128 \
#--nmb_prototypes 3000 3000 3000 \
#--epochs 400 \
#--batch_size 64 \
#--base_lr 1.2 \
#--final_lr 0.0012 \
#--freeze_prototypes_niters 300000 \
#--wd 0.000001 \
#--warmup_epochs 10 \
#--start_warmup 0.3 \
#--arch resnet50 \
#--sync_bn apex \
#--dump_path $EXPERIMENT_PATH \
#--fast_dev_run 5