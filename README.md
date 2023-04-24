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