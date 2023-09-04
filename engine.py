# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import datetime
import math
import time
from utils.misc import SmoothedValue
from utils.dist import (
    all_reduce_average,
    is_primary,
    barrier,
)


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch_classification(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_loader,
    logger,
):
    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    correct = 0
    total = 0

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        data = batch_data_label[0]  # Extract the data from batch_data_label
        labels = batch_data_label[1]  # Extract the labels from batch_data_label
        curr_time = time.time()
        curr_epoch_normalized = curr_epoch / args.max_epoch
        curr_lr = adjust_learning_rate(args, optimizer, curr_epoch_normalized)
        
        # Move data and labels to the appropriate device
        data = data.to(net_device)
        labels = labels.to(net_device)
        # Forward pass
        optimizer.zero_grad()
        
        inputs = data
        outputs = model(inputs.float())
        predicted_labels = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        loss_reduced = all_reduce_average(loss)
        loss_avg.update(loss_reduced.item())
        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()

        loss.backward()
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient, error_if_nonfinite=False)
        optimizer.step()

        time_delta.update(time.time() - curr_time)

        # Logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; Accuracy {100 * correct / total:0.2f}%; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            logger.log_scalars({"Train/loss": loss_avg.avg}, curr_iter, curr_epoch)
            logger.log_scalars({"Train/accuracy": 100 * correct / total}, curr_iter, curr_epoch)

        curr_iter += 1
        barrier()

    return correct / total



@torch.no_grad()
def evaluate_classification(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_loader,
    logger,
):
    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    correct = 0
    total = 0

    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        data = batch_data_label[0]  # Extract the data from batch_data_label
        labels = batch_data_label[1]  # Extract the labels from batch_data_label
        

        # Move data and labels to the appropriate device
        data = data.to(net_device)
        labels = labels.to(net_device)
        inputs = data
        outputs = model(inputs.float())
        # Compute loss
        loss = criterion(outputs, labels)
        loss_avg.update(loss.item())

        predicted_labels = torch.argmax(outputs, dim=1)
        print(labels)
        print(predicted_labels)
        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()

        time_delta.update(time.time() - curr_time)

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; Loss {loss_avg.avg:0.2f}; Accuracy {100 * correct / total:0.2f}%; Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            logger.log_scalars({"Eval/loss": loss_avg.avg}, curr_iter, curr_epoch)
            logger.log_scalars({"Eval/accuracy": 100 * correct / total}, curr_iter, curr_epoch)

        curr_iter += 1
        barrier()

    return correct / total
