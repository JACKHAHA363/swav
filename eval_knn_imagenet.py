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
import logging

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
    fix_random_seeds,
    MetricLogger
)
import src.resnet50 as resnet_models

logger = getLogger(__file__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")

#########################
#### main parameters ####
#########################
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--data_path", type=str, default="/datasets01/imagenet_full_size/061417/",
                    help="path to dataset repository")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--debug", action='store_true')
parser.add_argument("--nb_knn", default=[1, 5, 10, 20])
parser.add_argument("--temperature", default=0.07)

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default=None)
parser.add_argument("--fast_dev_run", default=0, type=int)

@torch.no_grad()
def extract_features(model, data_loader, fast_dev_run=0):
    metric_logger = MetricLogger(delimiter="  ")
    features = []
    count = 0
    labels = []
    for samples, lab in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        feats = model.forward_backbone(samples).clone()
        output_all = feats

        # update storage feature matrix
        features.append(output_all.cpu())
        labels.append(lab.cpu())
        count += 1
        if fast_dev_run > 0 and count > fast_dev_run:
            break

    print('Done with extracting features')
    return {'feats': torch.cat(features), 'labels': torch.cat(labels)}


def getknn(nb_knn, temperature, data_path, batch_size, model, fast_dev_run):
    # build data
    train_dataset = datasets.ImageFolder(os.path.join(data_path, "train"))
    val_dataset = datasets.ImageFolder(os.path.join(data_path, "val"))
    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )
    transform =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
    train_dataset.transform = transform
    val_dataset.transform = transform
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=12,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=12,
        pin_memory=True,
        shuffle=True,
    )
    logger.info("Building data done")

    ret = extract_features(model, train_loader, fast_dev_run)
    train_features = ret['feats']
    train_labels = ret['labels']
    ret = extract_features(model, val_loader, fast_dev_run)
    test_features = ret['feats']
    test_labels = ret['labels']
    print('Normalize...')
    train_features = torch.nn.functional.normalize(train_features, dim=1, p=2)
    test_features = torch.nn.functional.normalize(test_features, dim=1, p=2)
    result = collect_knn_results(train_features, train_labels, test_features,
                                 test_labels, nb_knn=nb_knn, temperature=temperature, 
                                 nb_classes=1000)
    return result


def main():
    args = parser.parse_args()
    fix_random_seeds(args.seed)

    # build model
    model = resnet_models.__dict__[args.arch](output_dim=0)
  
    # model to gpu
    model = model.cuda()
    model.eval()

    # load weights
    if args.pretrained is not None and os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cpu")
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
    result = getknn(args.nb_knn, args.temperature, args.data_path, args.batch_size,model, args.fast_dev_run)
    print(result)

if __name__ == "__main__":
    main()
