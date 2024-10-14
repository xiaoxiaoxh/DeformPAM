#!/bin/bash

# Customize
CONDA_ENV=DeformPAM
CONDA_ROOT=$(conda info --base)
set -e

# Stage 1: environtment creation
source $CONDA_ROOT/bin/activate
if [ -z $(conda env list | grep -v '#' | awk '{print $1}' | grep $CONDA_ENV) ]; then
conda create -n $CONDA_ENV python=3.8
fi
conda activate $CONDA_ENV

# Stage 2: Pytorch
echo "Installing Pytorch"
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Stage 3: OpenBlas-devel
echo "Installing OpenBlas-devel"
conda install -y openblas-devel -c anaconda -c conda-forge

# Stage 4.1: install a specific version of pip
echo "Installing pip 22.3.1"
pip install pip==22.3.1

# Stage 4.2: install setuptools
echo "Installing setuptools"
pip install setuptools==59.5.0

# Stage 4.3: MinkowskiEngine
if [ -z $(pip freeze | grep MinkowskiEngine) ]; then
echo "Installing MinkowskiEngine"
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
else
echo "MinkowskiEngine already installed"
fi

# Stage 5: Other Packages
echo "Installing other packages"
pip install -r requirements.txt