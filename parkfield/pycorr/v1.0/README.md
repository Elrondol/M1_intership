# PYCORR v1.0 #

Python3.7
Ambient seismic noise correlation package

# Basic philosophy
- retrieve seismic station information from FDSN catalog : http://service.iris.edu/irisws/fedcatalog/1/
- retrieve waveforms from FDSN-webservices and/or personal archive
- basic processing "on the fly": decimation, sensor response, gaps ...
- possible (pre-) processing before correlation
- xcorr with flexible parameters for optimized tomography and/or monitoring applications
- tensor rotation to retrieve RT information
- Basic toolbox to extract and plot correlations results from large output

# recommended install

wget https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh  
chmod u+x  Anaconda3-2019.03-Linux-x86_64.sh  
./Anaconda3-2019.03-Linux-x86_64.sh  

source .bashrc  
conda update -n base -c defaults conda  
conda create -n mypy3 python=3.7  
conda activate mypy3  
conda install h5py  
conda install ipdb  
conda install cartopy   
conda install statsmodels  
conda install scikit-learn  

conventional installation of obpsy :  
conda install obspy  

or MASTER version obspy:  
git clone https://github.com/obspy/obspy.git    
pip install /home/bouep/DATA/obspy  


set your python path 
export PYTHONPATH=".../pycorr/v1.0:$PYTHONPATH"


