# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from knn_utils import collect_knn_results

from src.utils import (
    initialize_exp,
    fix_random_seeds,
    MetricLogger
)
import src.resnet50 as resnet_models

logger = getLogger(__file__)


parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")

#########################
#### main parameters ####
#########################
parser.add_argument("--outfile", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--data_path", type=str, default="/data/home/lyuchen/swav_exp/new_stl10",
                    help="path to dataset repository")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--batch_size", default=256, type=int)

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")

class STL10withindex(datasets.STL10):
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return idx, img, label

@torch.no_grad()
def extract_features(model, data_loader):
    metric_logger = MetricLogger(delimiter="  ")
    features = None
    count = 0
    labels = None
    for index, samples, lab in metric_logger.log_every(data_loader, 10):
        import ipdb; ipdb.set_trace()
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = model.forward_backbone(samples).clone()

        features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
        labels = torch.zeros(len(data_loader.dataset)).long()
        print(f"Storing features into tensor of shape {features.shape}")
        index_all = index
        output_all = feats

        # update storage feature matrix
        features.index_copy_(0, index_all.cpu(), output_all.cpu())
        labels.index_copy_(0, index_all.cpu(), lab.cpu())
            
        count += samples.shape[0]
        if args.debug:
            break

    print('Done with extracting features')
    return {'feats': features, 'labels': labels}



def main():
    args = parser.parse_args()
    fix_random_seeds(args.seed)

    # build data
    train_dataset = STL10withindex(args.data_path, split='train')
    val_dataset = STL10withindex(args.data_path, split="test")
    tr_normalize = transforms.Normalize(
        mean = [0.43, 0.42, 0.39],
        std = [0.27, 0.26, 0.27]
    )
    train_dataset.transform = transforms.Compose([
        transforms.Resize(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_dataset.transform = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        tr_normalize,
    ])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    logger.info("Building data done")

    # build model
    model = resnet_models.__dict__[args.arch](output_dim=0, eval_mode=True)
  
    # model to gpu
    model = model.cuda()
    model.eval()

    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda:" + str(args.gpu_to_work_on))
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training with random weights")


    train_data = extract_features(model, train_loader)
    test_data = extract_features(model, val_loader)
    train_features = train_data['feats']
    test_features = test_data['feats']
    print('Normalize...')
    train_features = torch.nn.functional.normalize(train_features, dim=1, p=2)
    test_features = torch.nn.functional.normalize(test_features, dim=1, p=2)
    train_labels = train_data['labels']
    test_labels = test_data['labels']
    result = collect_knn_results(train_features, train_labels, test_features,
                                 test_labels, args.nb_knn, args.temperature,
                                 with_train=args.with_train)
    print(result)
    result.to_csv(args.outfile)

if __name__ == "__main__":
    main()
