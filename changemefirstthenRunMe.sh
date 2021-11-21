#!/bin/bash
#  Install the Intel® Distribution for Python, skip this if you have already installed the Intel® Distribution for Python,
#  otherwise do the following steps:


cd CNN_UCL_SINGELE_MODALITY  <-----root

conda env create -f cnn_run_conda_environment.yml

conda activate cnnrunenv


python  preprocess_inference_script.py


#   - singularity pull docker://kbronik/ms_cnn_ucl:latest  
# After running the above, a singularity image using docker hub (docker://kbronik/ms_cnn_ucl:latest) will be generated:

#  - path to singularity//..///ms_cnn_ucl_latest.sif  
#  singularity exec  ms_cnn_ucl-latest.simg  python (path to ...)/CNN_UCL_SINGELE_MODALITY/inference_script.py

python  inference_script.py

or?

singularity exec  ms_cnn_ucl-latest.simg  python (path to ...)/CNN_UCL_SINGELE_MODALITY/inference_script.py


echo "All calculation completed"

