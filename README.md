### final-project

## 1. Requirements

The code is tested with PyTorch 1.9.0, CUDA 10.2 and Python 3.6.

The necessary steps to get ready for the task are outlined below.

```
conda create --name gp1 python=3.6
conda activate gp1
conda install pytorch=1.9.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
cd third_party/pointnet2 && python setup.py install
pip install matplotlib
pip install plyfile
pip install 'trimesh>=2.35.39,<2.35.40'
pip install 'networkx>=2.2,<2.3'
pip install scipy
pip install torch_geometric
pip install torch_sparse
pip install torch_scatter
pip install path
```
## 2. Datasets

We use ShapeNet-55 for pre-training and ModelNet40 for fine-tuning.

See [README.md](./datasets/README.md) for details.

## 3. 3DTransNet Models

## 4. Pre-training

Pre-training with the default configuration, run the script:

```
./scripts/3dtransnet_pretrain.sh --CUDA <GPU>
```

## 5. Fine-tuning

You can start to fine-tune the pre-trained model and get 4 different models by:

```
./scripts/3dtransnet_base.sh --CUDA <GPU>
./scripts/3dtransnet_uf.sh --CUDA <GPU>
./scripts/3dtransnet_f.sh --CUDA <GPU>
./scripts/3dtransnet_ef.sh --CUDA <GPU>
```

## 6. Test

You can test one model by:

```
./scripts/3dtransnet_test.sh --CUDA <GPU> --test_ckpt <path/to/fine-tuned/model> --exp_name <exp_name>
```






