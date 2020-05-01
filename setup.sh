#! /bin/bash

# ensure that the script has been sourced rather than just executed
if [[ "${BASH_SOURCE[0]}" = "${0}" ]]; then
    echo "Please use 'source' to execute setup.sh!"
    exit 1
fi

CURRENTDIR=$(pwd)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

# First we setup the CUDA version to use 10.1
INSTALL_FOLDER="/usr/local"  # the location to look for CUDA installations at
TARGET_VERSION="10.1"        # the target CUDA version to switch to (if provided)

# check whether there is an installation of the requested CUDA version
if [[ ! -d "${INSTALL_FOLDER}/cuda-${TARGET_VERSION}" ]]; then
    echo "No installation of CUDA ${TARGET_VERSION} has been found!"
    set +e
    return
fi

# the path of the installation to use
cuda_path="${INSTALL_FOLDER}/cuda-${TARGET_VERSION}"

# filter out those CUDA entries from the PATH that are not needed anymore
path_elements=(${PATH//:/ })
new_path="${cuda_path}/bin"
for p in "${path_elements[@]}"; do
    if [[ ! ${p} =~ ^${INSTALL_FOLDER}/cuda ]]; then
        new_path="${new_path}:${p}"
    fi
done

# filter out those CUDA entries from the LD_LIBRARY_PATH that are not needed anymore
ld_path_elements=(${LD_LIBRARY_PATH//:/ })
new_ld_path="${cuda_path}/lib64:${cuda_path}/extras/CUPTI/lib64"
for p in "${ld_path_elements[@]}"; do
    if [[ ! ${p} =~ ^${INSTALL_FOLDER}/cuda ]]; then
        new_ld_path="${new_ld_path}:${p}"
    fi
done

# update environment variables
export CUDA_HOME="${cuda_path}"
export CUDA_ROOT="${cuda_path}"
export LD_LIBRARY_PATH="${new_ld_path}"
export PATH="${new_path}"

echo "Switched to CUDA ${TARGET_VERSION}."

if [[ -d "./env/conda/envs/chips/" ]]; then
    source env/conda/bin/activate
    conda activate chips
else
    cd env/  # Go to the env directory

    # Download the latest version of miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh --no-check-certificate

    # Install miniconda3 in the current directory
    bash miniconda.sh -b -p $DIR/env/conda
    rm miniconda.sh

    # Activate miniconda and create the chips environment
    source conda/bin/activate
    conda update -n base -c defaults conda -y
    conda config --add envs_dirs $DIR/env/conda/envs
    conda config --add envs_dirs $DIR/env/conda/envs
    conda env create -f $DIR/env/environment.yaml

    # Clean the miniconda install
    conda clean --all -y

    # Make sure the base environement is not enabled by default
    conda config --set auto_activate_base false

    conda activate chips
fi

# Go back to the user directory
cd $CURRENTDIR

if [[ -f ".comet" ]]; then
    while read LINE; do export "$LINE"; done < .comet
fi

alias run="python $DIR/scripts/run.py"
alias preprocess="python $DIR/scripts/preprocess.py"
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Setup complete."
return
