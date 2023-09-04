# Dataset

## ShapeNet-55

Please download the ShapeNet55 pointcloud data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). Store it in `./datasets/shapenet`.

Pre-process the data:

```
python ./datasets/pre_data.py
```

## ModelNet40

Download and pre-process the data:

```
python ./datasets/modelnet.py
python ./datasets/finetune_data.py
```

## Folder Structure

```
│datasets/
├──modelnet/
│  ├── raw
│  │  ├── airplane
│  │  ├── bathtub
│  │  ├── ...
│  ├── classes.pth
│  ├── train_dataset.pth
│  ├── valid_dataset.pth
├──shapenet/
│  ├── ShapeNet55/shapenet_pc
│  │  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│  │  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
│  │  ├── ...
│  ├── class_mapping.txt
│  ├── test.txt
│  ├── train.txt
│  ├── val.txt
├──finetune_data.py
├──modelnet.py
├──pre_data.py
├──shapenet.py
```






