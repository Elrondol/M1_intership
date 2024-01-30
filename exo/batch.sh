#!/bin/bash

#OAR -n python

#OAR --project iste-equ-ondes 

#OAR -l /nodes=4/core=5,walltime=24:00:00 

source /data/ondes/parisnic/miniconda3/bin/activate
conda activate py3

# path to use the python
export PATH="/data/ondes/parisnic/miniconda3/bins/:$PATH"

python corr_2wavev2.py
