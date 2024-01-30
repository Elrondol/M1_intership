#!/bin/bash

#OAR -n python

#OAR --project iste-equ-ondes 

#OAR -l /nodes=1/core=1,walltime=8:00:00 

source /data/ondes/parisnic/miniconda3/bin/activate
conda activate py3

# path to use the python
export PATH="/applis/environments/conda.sh:$PATH"
export PYTHONPATH="/data/projects/faultscan/user/parisnic/m1_internship/parkfield/pycorr/v1.0:$PYTHONPATH"

python 02_download_data.py
