#!/bin/bash

# Set your variables
CUDA='0'
exp_name='modelnet40_earlystopping'
finetune=false
earlystopping=true
freeze=false
checkpoint_name='checkpoint_modelnet40_earlystopping.pth'
checkpoint_best_name='checkpoint_modelnet40_best_earlystopping.pth'
test_ckpt='checkpoints_cls/checkpoint_modelnet40_best_earlystopping.pth'

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

