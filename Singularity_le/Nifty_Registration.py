from __future__ import print_function, division

import logging
import os
import signal
import time
import numpy as np
import shutil
# import SimpleITK as sitk
# from matplotlib import pyplot
# import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

read_file_path = './Train/'
read_folder_list = os.listdir(read_file_path)

for ii in range(0, len(read_folder_list)):

    read_folder_name = read_folder_list[ii]
    print(read_folder_name)
    folder_path = os.path.join(read_file_path, read_folder_name)

    os.chdir(folder_path)
    os.system('/usr/local/fsl/bin/flirt -in Flair.nii -ref ./MNI152_T1_1mm.nii.gz -out Flair_reg.nii.gz -omat Flair_reg.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear')

    os.system('/usr/local/fsl/bin/bet Flair_reg Flair_reg_brain  -f 0.5 -g 0')
    
    os.system('python ./N4BiasFieldCorrection-master/N4BiasFieldCorrection_FL.py')
    os.system('rm Flair_reg_brain_bias_mask.nii.gz')
   
  