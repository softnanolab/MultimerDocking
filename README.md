Work in progress



## Installation
Run this to add the submodules first:
```
git submodule update --init --recursive
```
With uv and conda env:
```
git clone https://github.com/softnanolab/MultimerDocking.git
cd MultimerDocking
conda create -n dock python=3.12
conda activate dock
uv pip install -e ".[dev]"
cd submodules/mint && uv pip install -e .   # this is to install mint
```


With uv and uv env:
```
git clone https://github.com/softnanolab/MultimerDocking.git
cd MultimerDocking
uv sync
```
