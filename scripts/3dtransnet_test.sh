#!/bin/bash

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --CUDA)
        CUDA="$2"
        shift
        shift
        ;;
        --test_ckpt)
        test_ckpt="$2"
        shift
        shift
        ;;
        --exp_name)
        exp_name="$2"
        shift
        shift
        ;;
        *)
        shift
        ;;
    esac
done

# 设置默认参数
if [ -z "$CUDA" ]; then
    CUDA=0
fi

if [ -z "$test_ckpt" ]; then
    test_ckpt="checkpoints_cls/checkpoint_finetune_best_no_earlystopping.pth"
fi

if [ -z "$exp_name" ]; then
    exp_name='main'
fi

# 运行程序
python /home/uceeuam/graduation_project/test_cls.py --CUDA "$CUDA" --test_ckpt "$test_ckpt" --exp_name "$exp_name"
