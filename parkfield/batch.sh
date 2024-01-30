#!/bin/bash

#OAR -n python

#OAR --project faultscan

#OAR -l /nodes=1/core=1,walltime=12:00:00 

source /applis/environments/conda.sh
conda activate py3

# path to use the python
export PATH="/applis/environments/conda.sh:$PATH"
export PYTHONPATH="pycorr/v1.0:$PYTHONPATH"

#python plot_ppsds.py
python ponder_july.py
