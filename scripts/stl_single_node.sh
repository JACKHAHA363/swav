# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --job-name=deepclusterv2_400ep_2x224_pretrain
#SBATCH --time=25:00:00
#SBATCH --mem=64G

DATASET_PATH="/data/home/lyuchen/swav_exp/new_stl10"
EXPERIMENT_PATH="./experiments/stl/deepclusterv2_400ep_2x224_pretrain"
mkdir -p $EXPERIMENT_PATH

CMD="python -m torch.distributed.launch --nnodes 1 --nproc_per_node 1 \
main_deepclusterv2.py \
--data_path $DATASET_PATH \
--nmb_crops 2 \
--size_crops 96 \
--min_scale_crops 0.33 \
--max_scale_crops 1. \
--crops_for_assign 0 1 \
--temperature 0.1 \
--feat_dim 128 \
--nmb_prototypes 512 \
--epochs 400 \
--batch_size 64 \
--base_lr 4.8 \
--final_lr 0.0048 \
--freeze_prototypes_niters 300000 \
--wd 0.000001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--arch resnet50 \
--sync_bn pytorch \
--dump_path $EXPERIMENT_PATH"

echo $CMD
$CMD
#srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label $CMD


