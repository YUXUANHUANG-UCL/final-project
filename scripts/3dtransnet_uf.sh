#!/bin/bash

# Set your variables
CUDA='0'
exp_name='finetune_no_earlystopping_no_freezing'
finetune=true
earlystopping=false
freeze=false
checkpoint_name='checkpoint_finetune_no_earlystopping_no_freezing.pth'
checkpoint_best_name='checkpoint_finetune_best_no_earlystopping_no_freezing.pth'
test_ckpt='checkpoints_cls/checkpoint_finetune_best_no_earlystopping_no_freezing.pth'

# 处理命令行参数
for arg in "$@"; do
    case "$arg" in
        --CUDA)
            CUDA="$2"
            shift
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Run your script with the specified parameters
python /home/uceeuam/graduation_project/classification_finetune_modelnet40.py \
    --CUDA "$CUDA" \
    --exp_name "$exp_name" \
    --finetune "$finetune" \
    --earlystopping "$earlystopping" \
    --freeze "$freeze" \
    --checkpoint_name "$checkpoint_name" \
    --checkpoint_best_name "$checkpoint_best_name" \
    --test_ckpt "$test_ckpt"
