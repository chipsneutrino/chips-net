#! /bin/bash

CURRENTDIR=$(pwd)

# If we don't have the deps directory first make it and then cd into it
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

if [ -d "env/miniconda/envs/chips-cvn/" ]
then
    echo "Conda env installed"
    source env/miniconda/bin/activate
    conda activate chips-cvn
    conda env update --file $DIR/env/environment.yml
else
    # Go to the env directory
    cd env/

    # Download the latest version of miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh --no-check-certificate

    # Install miniconda3 in the current directory
    bash miniconda.sh -b -p $DIR/env/miniconda
    rm miniconda.sh

    # Activate miniconda and create the chips-cvn environment
    source miniconda/bin/activate
    conda update -n base -c defaults conda
    conda config --add envs_dirs $DIR/env/miniconda/envs
    conda config --add envs_dirs $DIR/env/miniconda/envs
    conda env create -f $DIR/env/environment.yml

    # Clean the miniconda install
    conda clean --all -y

    # Make sure the base environement is not enabled by default
    conda config --set auto_activate_base false

    conda activate chips-cvn
fi

echo "chips-cvn setup done"

# Go back to the user directory
cd $CURRENTDIR



