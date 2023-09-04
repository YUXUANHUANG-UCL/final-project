#!/bin/bash
# 设置默认参数
CUDA=0
exp_name="pretrain"

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

# 运行Python脚本
python /home/uceeuam/graduation_project/classification_pretrain.py --CUDA "$CUDA" --exp_name "$exp_name"
