# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch.utils.data import DataLoader

from models.model_3detr_classification import build_3detr
from datasets.shapenet import ShapeNetDataset
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=2, type=int)
    parser.add_argument("--enc_dim", default=1024, type=int)
    parser.add_argument("--enc_ffn_dim", default=512, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)
    
    ### Decoder
    parser.add_argument("--dec_nlayers", default=0, type=int)
    parser.add_argument("--dec_dim", default=1024, type=int)
    parser.add_argument("--dec_ffn_dim", default=1024, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=1024, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=1024, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")

    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)
    parser.add_argument("--num_classes", default=55, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--max_epoch", default=120, type=int)
    parser.add_argument("--eval_every_epoch", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--finetune", default=False)
    parser.add_argument("--pretrain", default=True)
    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--test_ckpt", default='checkpoints_pretrain/checkpoint_best.pth', type=str)

    ##### I/O #####
    parser.add_argument("--CUDA", default=7, type=int)
    parser.add_argument("--checkpoint_dir", default='checkpoints_pretrain', type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=10, type=int)
    parser.add_argument("--use_enc_only", default=True)
    parser.add_argument("--exp_name", default='main', type=str)
    
    return parser


# 3DETR codebase specific imports
from engine import evaluate_classification, train_one_epoch_classification
from optimizer import build_optimizer
from utils.dist import is_primary
from utils.io import save_checkpoint
from utils.logger import Logger


def train_classification_model(
    args,
    model,
    optimizer,
    criterion,
    dataloaders,
    best_val_metrics,
):
    """
    Main training loop for classification model.
    This trains the model for `args.max_epoch` epochs, evaluates the model on the validation set,
    and performs the final evaluation on the test set.
    """

    num_iters_per_epoch = len(dataloaders["train"])
    num_iters_per_val_epoch = len(dataloaders["val"])
    num_iters_per_test_epoch = len(dataloaders["test"])
    print(f"Model: {model}")
    print(f"Training started at epoch {args.start_epoch} until {args.max_epoch}.")
    print(f"One training epoch = {num_iters_per_epoch} iters.")
    print(f"One validation epoch = {num_iters_per_val_epoch} iters.")
    print(f"One test epoch = {num_iters_per_test_epoch} iters.")
    logger = Logger(args.checkpoint_dir, args.exp_name)
    for epoch in range(args.start_epoch, args.max_epoch):

        train_metrics = train_one_epoch_classification(
            args,
            epoch,
            model,
            optimizer,
            criterion,
            dataloaders["train"],
            logger,
        )

        # Save checkpoint
        save_checkpoint(
            args.checkpoint_dir,
            model,
            optimizer,
            epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )

        # Evaluate on the validation set
        if epoch % args.eval_every_epoch == 0 or epoch == (args.max_epoch - 1):
            val_metrics = evaluate_classification(
                args,
                epoch,
                model,
                criterion,
                dataloaders["val"],
                logger,
            )

            # Update best validation metrics if necessary
            if not best_val_metrics or val_metrics > best_val_metrics:
                best_val_metrics = val_metrics
                save_checkpoint(
                    args.checkpoint_dir,
                    model,
                    optimizer,
                    epoch,
                    args,
                    best_val_metrics,
                    filename="checkpoint_best.pth",
                )
                print(f"Epoch [{epoch}/{args.max_epoch}] saved current best validation checkpoint. Accuracy:{best_val_metrics}")

        # Print training metrics
        print("==" * 10)
        print(f"Epoch [{epoch}/{args.max_epoch}]; Train Metrics: {train_metrics}")
        print("==" * 10)

    print("Training Finished.")

    # Evaluate on the test set
    final_metrics = evaluate_classification(
        args,
        args.max_epoch - 1,
        model,
        criterion,
        dataloaders["test"],
        logger,
    )
    print("==" * 10)
    print(f"Final Metrics on Test Set: {final_metrics}")
    print("==" * 10)

def test_model(args, model, criterion, dataset_loader):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        print(f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}")
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model.load_state_dict(sd["model"])
    logger = Logger(args.checkpoint_dir, args.exp_name)
    # criterion = None  # Do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    accuracy = evaluate_classification(args, epoch, model, criterion, dataset_loader, logger)
    if is_primary():
        print("==" * 10)
        print(f"Test model; Accuracy {accuracy * 100:0.2f}%")
        print("==" * 10)


def main(args):
    print(f"Called with args: {args}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    data_dir = "datasets/shapenet/ShapeNet55/shapenet_pc"
    train_label_file = 'datasets/shapenet/train.txt'
    val_label_file = 'datasets/shapenet/val.txt'
    test_label_file = 'datasets/shapenet/test.txt'

    train_dataset = ShapeNetDataset(data_dir, train_label_file)
    val_dataset = ShapeNetDataset(data_dir, val_label_file)
    test_dataset = ShapeNetDataset(data_dir, test_label_file)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_3detr(args)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    best_val_metrics = 0

    train_classification_model(args, model, optimizer, criterion, dataloaders, best_val_metrics)

    test_model(args, model, criterion, test_loader)
    
if __name__ == "__main__":
    args = make_args_parser().parse_args()
    torch.cuda.set_device(args.CUDA)
    main(args)
