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
git clone https://github.com/dmn-sjk/shift-dev.git
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
export PYTHONPATH=$PYTHONPATH:[clad_repository_folder]
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
To generate result charts use `evaluation/generate_results.py` script.
