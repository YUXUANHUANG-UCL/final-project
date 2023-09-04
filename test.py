import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model_3detr_classification import build_preencoder, build_encoder, build_3detr, ClassificationModel
from datasets.shapenet import ShapeNetDataset
import argparse
import os
import sys
import pickle
import numpy as np
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler

from classification_pretrain import make_args_parser, test_model

def main(args):
    print(f"Called with args: {args}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    num_classes = 55
    data_dir = "/home/uceeuam/graduation_project/datasets/shapenet/ShapeNet55/shapenet_pc"
    test_label_file = '/home/uceeuam/graduation_project/datasets/shapenet/test.txt'

    test_dataset = ShapeNetDataset(data_dir, test_label_file)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)

    model = build_3detr(args)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    test_model(args, model, criterion, test_loader)

if __name__ == "__main__":
    args = make_args_parser().parse_args()
    torch.cuda.set_device(8)  # Set the target GPU device
    main(args)
