#!/bin/bash

#OAR -n python

#OAR --project iste-equ-ondes

#OAR -l /nodes=1/core=8,walltime=12:00:00 

source /data/ondes/parisnic/miniconda3/bin/activate
conda activate py3

# path to use the python
export PATH="/data/ondes/parisnic/miniconda3/bin/activate:$PATH"
export PYTHONPATH="pycorr/v1.0:$PYTHONPATH"

python plot_ppsds.py
