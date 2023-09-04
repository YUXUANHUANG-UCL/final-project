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
    if curr_epoch < 4/120 and args.finetune:
        curr_lr = curr_lr * 10
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
        data = batch_data_label['pointcloud']  # Extract the data from batch_data_label
        labels = batch_data_label['category']  # Extract the labels from batch_data_label
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
        data = batch_data_label['pointcloud']  # Extract the data from batch_data_label
        labels = batch_data_label['category']  # Extract the labels from batch_data_label
        

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

@torch.no_grad()
def test_classification(
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

    tp = torch.zeros(args.num_classes_new, device=net_device)
    fp = torch.zeros(args.num_classes_new, device=net_device)
    tn = torch.zeros(args.num_classes_new, device=net_device)
    fn = torch.zeros(args.num_classes_new, device=net_device)
    class_counts = torch.zeros(args.num_classes_new, device=net_device)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        data = batch_data_label['pointcloud']
        labels = batch_data_label['category']

        data = data.to(net_device)
        labels = labels.to(net_device)
        inputs = data
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss_avg.update(loss.item())

        predicted_labels = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()
        for t, p in zip(labels.view(-1), predicted_labels.view(-1)):
            if t == p:
                tp[t] += 1
                for i in range(args.num_classes_new):
                    if i != t:
                        tn[i] += 1
            else:
                fp[p] += 1
                fn[t] += 1
            class_counts[t] += 1
        '''
        for i in range(args.num_classes_new):
            tn[i] = class_counts.sum() - tp.sum() - fp.sum() - fn[i]
        '''
        time_delta.update(time.time() - curr_time)

        if is_primary() and (curr_iter % args.log_every == 0 or curr_iter == num_batches - 1):
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; Loss {loss_avg.avg:0.2f}; Accuracy {100 * correct / total:0.2f}%; Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            logger.log_scalars({"Eval/loss": loss_avg.avg}, curr_iter, curr_epoch)
            logger.log_scalars({"Eval/accuracy": 100 * correct / total}, curr_iter, curr_epoch)

        curr_iter += 1
        barrier()
    # Calculate metrics for each class
    f1_scores = []
    accuracies = []
    precisions = []
    sensitivities = []
    specificities = []
    for i in range(args.num_classes_new):
        precision = tp[i] / (tp[i] + fp[i])
        recall = tp[i] / (tp[i] + fn[i])
        specificity = tn[i]/(args.num_classes_new-1) / (tn[i]/(args.num_classes_new-1) + fp[i])
        sensitivity = recall
        accuracy = (tp[i] + tn[i]/(args.num_classes_new-1)) / (tp[i] + tn[i]/(args.num_classes_new-1) + fp[i] + fn[i])
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores.append(f1)
        accuracies.append(accuracy)
        precisions.append(precision)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

        # Output metrics for each class
        print(f"Class {i} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, F1 Score: {f1:.4f}")
        
        # Log metrics for each class
        logger.log_scalars({f"Eval/Class_{i}/Accuracy": accuracy}, curr_epoch, curr_epoch)
        logger.log_scalars({f"Eval/Class_{i}/Precision": precision}, curr_epoch, curr_epoch)
        logger.log_scalars({f"Eval/Class_{i}/Sensitivity": sensitivity}, curr_epoch, curr_epoch)
        logger.log_scalars({f"Eval/Class_{i}/Specificity": specificity}, curr_epoch, curr_epoch)
        logger.log_scalars({f"Eval/Class_{i}/F1_Score": f1}, curr_epoch, curr_epoch)

    # Find the indices of the minimum values in each list
    min_f1_index = f1_scores.index(min(f1_scores))
    min_accuracy_index = accuracies.index(min(accuracies))
    min_precision_index = precisions.index(min(precisions))
    min_sensitivity_index = sensitivities.index(min(sensitivities))
    min_specificity_index = specificities.index(min(specificities))

    # Create a dictionary to hold the metric names and their corresponding indices
    metric_indices = {
        "F1 Score": [f1_scores[min_f1_index], min_f1_index],
        "Accuracy": [accuracies[min_accuracy_index], min_accuracy_index],
        "Precision": [precisions[min_precision_index], min_precision_index],
        "Sensitivity": [sensitivities[min_sensitivity_index], min_sensitivity_index],
        "Specificity": [specificities[min_specificity_index], min_specificity_index]
    }

    # Log the minimum values and their corresponding indices
    for metric_name, index in metric_indices.items():
        logger.log_scalars({f"Worst {metric_name}, Class {index[1]}": {index[0]}}, curr_epoch, curr_epoch)
    
    # Calculate and output average values with weighted class proportions
    class_proportions = class_counts.float() / class_counts.float().sum()
    # Calculate weighted averages
    avg_accuracy = sum(prop * acc for prop, acc in zip(class_proportions, accuracies))
    avg_precision = tp.sum() / (tp.sum() + fp.sum())
    avg_sensitivity = sum(prop * spec for prop, spec in zip(class_proportions, sensitivities))
    avg_specificity = sum(prop * spec for prop, spec in zip(class_proportions, specificities))
    avg_f1 = torch.stack(f1_scores).sum() / len(f1_scores)

    print(f"Average - Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Sensitivity: {avg_sensitivity:.4f}, Specificity: {avg_specificity:.4f}, F1 Score: {avg_f1:.4f}")
    
    # Log average metrics
    logger.log_scalars({"Eval/Avg_Accuracy": avg_accuracy}, curr_epoch, curr_epoch)
    logger.log_scalars({"Eval/Avg_Precision": avg_precision}, curr_epoch, curr_epoch)
    logger.log_scalars({"Eval/Avg_Sensitivity": avg_sensitivity}, curr_epoch, curr_epoch)
    logger.log_scalars({"Eval/Avg_Specificity": avg_specificity}, curr_epoch, curr_epoch)
    logger.log_scalars({"Eval/Avg_F1_Score": avg_f1}, curr_epoch, curr_epoch)

    return correct / total
