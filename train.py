"""Train script.

Usage:
    train.py <hparams> <dataset> <dataset_root>
"""
import os
import vision
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.trainer import Trainer
from glow.config import JsonConfig
import random
import torch
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset = args["<dataset>"]
    dataset_root = args["<dataset_root>"]
    assert dataset in vision.Datasets, (
        "`{}` is not supported, use `{}`".format(dataset, vision.Datasets.keys()))
    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    hparams = JsonConfig(hparams)
    dataset = vision.Datasets[dataset]
    # set transform of dataset
    transform = transforms.Compose([
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.Resize(hparams.Data.resize),
        transforms.ToTensor()])
    # fix random seed
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(30)
    # build graph and dataset
    built = build(hparams, True)
    dataset = dataset(dataset_root, transform=transform)
    # begin to train
    trainer = Trainer(**built, dataset=dataset, hparams=hparams)
    trainer.train()
