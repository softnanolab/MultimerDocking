Work in progress

## Installation
With uv and conda env:
```
git clone https://github.com/softnanolab/MultimerDocking.git
cd MultimerDocking
git submodule update --init --recursive
conda create -n dock python=3.12
conda activate dock
uv pip install -e ".[dev]"
cd submodules/mint && uv pip install -e .   # this is to install mint
```

For later submodule updates from the parent repo:
```
git submodule update --remote --merge
```
Then commit and push the updated submodule SHA in the parent repo if you want to record it.

Alternatively, you can `git pull` inside the submodule and then commit the updated submodule SHA in the parent repo.


With uv and uv env:
```
git clone https://github.com/softnanolab/MultimerDocking.git
cd MultimerDocking
git submodule update --init --recursive
uv sync
cd submodules/mint && uv pip install -e .
```
