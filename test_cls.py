import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model_3detr_classification import build_3detr
import numpy as np
import torch
from torch.utils.data import DataLoader

from classification_finetune_modelnet40 import make_args_parser, test_model
from datasets.finetune_data import load_finetune_modelnet40

def main(args):
    print(f"Called with args: {args}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    _, val_dataset = load_finetune_modelnet40()

    test_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_3detr(args)
    in_features = model.fc3.in_features
    model.fc3 = nn.Linear(in_features, args.num_classes_new)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    
    test_model(args, model, criterion, test_loader)
    '''
    for batch_data_label in test_loader:
        print(batch_data_label['category'][0])
        print(batch_data_label['category'][0].shape)
        break  # 这将只获取第一个批次数据
    '''
if __name__ == "__main__":
    args = make_args_parser().parse_args()
    torch.cuda.set_device(args.CUDA)  # Set the target GPU device
    main(args)
