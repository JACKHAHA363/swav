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
import random
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
parser.add_argument("--debug", action='store_true')
parser.add_argument("--nb_knn", default=[1, 5, 10, 20])
parser.add_argument("--temperature", default=0.1)

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default=None)
class STL10withindex(datasets.STL10):
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return idx, img, label

@torch.no_grad()
def extract_features(args, model, data_loader):
    metric_logger = MetricLogger(delimiter="  ")
    features = None
    count = 0
    labels = None
    for index, samples, lab in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = model.forward_backbone(samples).clone()

        if features is None:
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
    tr_normalize = transforms.Normalize(
        mean = [0.43, 0.42, 0.39],
        std = [0.27, 0.26, 0.27]
    )
    transform = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        tr_normalize,
    ])
    dataset = STL10withindex(args.data_path, split='train', transform=transform)
    allids = [i for i in range(len(dataset))]
    random.shuffle(allids)
    split = int(0.9 * len(allids))
    train_ids = allids[:split]
    test_ids = allids[split:]
    data_loader = torch.utils.data.DataLoader(
        dataset,
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
    if args.pretrained is not None and os.path.isfile(args.pretrained):
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


    features = extract_features(args, model, data_loader)
    train_features = features['feats'][train_ids]
    test_features = features['feats'][test_ids]
    print('Normalize...')
    train_features = torch.nn.functional.normalize(train_features, dim=1, p=2)
    test_features = torch.nn.functional.normalize(test_features, dim=1, p=2)
    train_labels = features['labels'][train_ids]
    test_labels = features['labels'][test_ids]
    result = collect_knn_results(train_features, train_labels, test_features,
                                 test_labels, args.nb_knn, args.temperature)
    print(result)
    #result.to_csv(args.outfile)

if __name__ == "__main__":
    main()