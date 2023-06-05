# Continual adaptation

## Dependencies setup:
1. Clone the repository
2. Create conda environment from an environment.yml file:

```bash
cd continual_adaptation
conda env create -f environment.yml
```

3. Activate conda env:

```bash
conda activate conti_adapt
```

3. Clone fork of shift-dev repository to your desired location:

```bash
git clone -b domain_selection https://github.com/dmn-sjk/shift-dev.git
```

4. Install shift-dev package with symbolic links in case of modifications

```bash 
cd shift-dev
pip install -e .
```

5. Clone fork of CLAD repository to your desired location:

```bash
git clone https://github.com/dmn-sjk/CLAD.git
```

6. Add CLAD repository to python path:

```bash
export PYTHONPATH=$PYTHONPATH:<clad_repository_folder>
```

## Datasets setup:
### SHIFT
1. Go to cloned shift-dev repository from [Dependencies setup section](#dependencies-setup) and download the dataset with the following command:

```bash
cd shift-dev
python download.py --view front --group "[img, det_2d]" --split all --framerate images <TARGET_DIR>
```

### CLAD
1. Download and decompress labeled SODA10M dataset following the instructions from the [website](https://soda-2d.github.io/download.html#instructions).


### CIFAR10 and CIFAR10C
1. Download and decompress CIFAR10 dataset:
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
```

2. Download and decompress CIFAR10C dataset:
```bash
wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar
```
3. Use `process_cifar.py` script from `utils` folder preprocess the cifar datasets:
```bash
python utils/process_cifar.py cifar-10c <path_to_CIFAR10C_dataset> <path_to_CIFAR10_dataset>
```

## Example source train:
```bash
python train_source.py --data_root /raid/NFS_SHARE/datasets --dataset shift --cuda 0 --num_workers 5 --run_name source_size32 --scheduler_gamma 0.85 --model wideresnet28 --wandb
```


## Example tta test:
```bash
python test_adaptation.py --data_root /raid/NFS_SHARE/datasets --save_results --method cotta --dataset shift --benchmark shift_mix --cuda 0 --num_workers 5 --run_name sgd_clear --pretrained_model_path models_checkpoints/shift_wideresnet28_source_size32.pth --model wideresnet28 --batch_size 10
```

## Evaluation
To generate result charts use `evaluation/generate_results.py` script. Example usage:
```bash
python evaluation/generate_results.py --results_name cifar10c_standard_resnet50_clear --logs_regex cifar*unifoptim cifar10c_cifar10c_repetitive_frozen_wideresnet28_clear
```
