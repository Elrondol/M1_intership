#!/bin/bash

#OAR -n python

#OAR --project faultscan

#OAR -l /nodes=1/core=1,walltime=48:00:00 

source /applis/environments/conda.sh
conda activate py3

# path to use the python
export PATH="/applis/environments/conda.sh:$PATH"

#python run_beam_daily_TR_passive.py
#python run_backpropagation_passive.py
python run_gridsearch_noise_source_passive.py