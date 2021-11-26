#!/usr/bin/env python3.7


# --------------------------------------------------
#
#     Copyright (C) {2021}  {Le Zhang and Kevin Bronik}
#
#     UCL Medical Physics and Biomedical Engineering
#     https://www.ucl.ac.uk/medical-physics-biomedical-engineering/
#     UCL Queen Square Institute of Neurology
#     https://www.ucl.ac.uk/ion/

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#
#     {Multiple sclerosis Lesion Segmentation}  Copyright (C) {2021}
#     This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#     This is free software, and you are welcome to redistribute it
#     under certain conditions; type `show c' for details.
from __future__ import print_function, division
import signal
import argparse
import os
import shutil
import subprocess
import textwrap
import tempfile
# import SimpleITK as sitk
import logging
import os
import time
import sys
import numpy as np
import shutil
import platform
from sources.base import test_cascaded_model
from sources.build_model import cascade_model
from sources.postprocess import invert_registration
from sources.preprocess import preprocess_scan
import os
import signal
import time
import shutil
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sources.nets import get_network
from keras import optimizers, losses
import tensorflow as tf
from numpy import inf
from keras  import backend as K
K.set_image_data_format('channels_first')
from sources.load_options import load_options, print_options

CEND      = '\33[0m'
CBOLD     = '\33[1m'
CITALIC   = '\33[3m'
CURL      = '\33[4m'
CBLINK    = '\33[5m'
CBLINK2   = '\33[6m'
CSELECTED = '\33[7m'

CBLACK  = '\33[30m'
CRED    = '\33[31m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE  = '\33[36m'
CWHITE  = '\33[37m'

CBLACKBG  = '\33[40m'
CREDBG    = '\33[41m'
CGREENBG  = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG   = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG  = '\33[46m'
CWHITEBG  = '\33[47m'

CGREY    = '\33[90m'
CRED2    = '\33[91m'
CGREEN2  = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2   = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2  = '\33[96m'
CWHITE2  = '\33[97m'

CGREYBG    = '\33[100m'
CREDBG2    = '\33[101m'
CGREENBG2  = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2   = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2  = '\33[106m'
CWHITEBG2  = '\33[107m'
# singularity exec /media/le/Disk1/nicmslesions3.sif nicwrap.py
# --weights1 /home/le/Desktop/Only_Flair_CNN_model/CNN_UCL_SINGELE_MODALITY/nets/Model_trained/nets/model_1.hdf5
# --weights2 /home/le/Desktop/Only_Flair_CNN_model/CNN_UCL_SINGELE_MODALITY/nets/Model_trained/nets/model_2.hdf5
# --flair /home/le/Desktop/Singularity_le/00015824_20111213/Flair_reg.nii.gz
# --lesions /home/le/Desktop/Singularity_le/00015824_20111213/output.nii.gz
# --no-register
# --batch-size 128
# --threshold 0.5
# --bias
print("##################################################")
print('\x1b[6;30;45m' + 'Multiple sclerosis Lesion Segmentation    ' + '\x1b[0m')
print('\x1b[6;30;45m' + 'Medical Physics and Biomedical Engineering' + '\x1b[0m')
print('\x1b[6;30;45m' + 'UCL - 2021                                ' + '\x1b[0m')
print('\x1b[6;30;45m' + 'Le Zhang and Kevin Bronik                 ' + '\x1b[0m')
print("##################################################")
parser = argparse.ArgumentParser(description="MS Lesion Segmentation using singularity/docker.")
# parser.add_argument("-w1", "--weights1", default="/nets/Model_trained/nets/model_1.hdf5")
# parser.add_argument("-w2", "--weights2", default="/nets/Model_trained/nets/model_2.hdf5")
# parser.add_argument("-pf", "--path_to_test_folders", required=True)
# parser.add_argument("-fn", "--flair_name", required=True)
# parser.add_argument("-l", "--lesions")
# parser.add_argument("-bz", "--batch-size", dest="batch_size", default="256")
# parser.add_argument("-b", "--bias")
# parser.add_argument("-r", "--register")
# parser.add_argument("-s", "--skull_strip")
# parser.add_argument("-t", "--threshold", default="0.99")
# parser.add_argument("-fsl", "--fsl")

parser.add_argument("--weights1", action="store_true")
parser.add_argument("--weights2", action="store_true")
parser.add_argument("--path_to_test_folders", required=True)
parser.add_argument("--flair_name", required=True, default="flair")
parser.add_argument("--lesions", action="store_true")
parser.add_argument("--batch_size",action="store_true", default="256")
parser.add_argument("--bias", action="store_true")
parser.add_argument("--register", action="store_true")
parser.add_argument("--skull_strip", action="store_true")
parser.add_argument("--threshold", action="store_true", default="0.5")
parser.add_argument("--N4", action="store_true")
parser.add_argument("--rf", action="store_true")
parser.add_argument("--reg_space", action="store_true")

args = parser.parse_args()
print(args)

config_template = """
[database]
train_folder = /path/to/training
inference_folder = /path/to/inference
tensorboard_folder = /path/to/tensorboardlogs
flair_tags = Flair
roi_tags = None
register_modalities = False
bias_correction = False
batch_prediction = False
reg_space = FlairtoT1
denoise = False
denoise_iter = 3
bias_iter = 10
bias_smooth = 20
bias_type = 1
bias_choice = All
skull_stripping = False
save_tmp = True
debug = True

[train]
full_train = True
pretrained_model = None
balanced_training = True
fraction_negatives = 2.0

[model]
name = Model_trained
pretrained = None
train_split = 0.25
max_epochs = 2000
patience = 50
batch_size = 128
net_verbose = 1
gpu_number = 0

[tensorboard]
port = 8080

[postprocessing]
t_bin = 0.5
l_min = 10
min_error = 0.5
"""
options = {}
# shutil.copyfile(args.weights1, os.path.join(weights_dir, "model_1.hdf5"))
# shutil.copyfile(args.weights2, os.path.join(weights_dir, "model_2.hdf5"))
options['experiment'] = 'Model_trained_newtry'
options['train_folder'] = '/path/to/training'
# options['test_folder'] = '/path/to/inference'
options['test_folder'] = os.path.join(args.path_to_test_folders)
options['output_folder'] = '/output'
options['current_scan'] = 'scan'
# options['t1_name'] = default_config.get('database', 't1_name')
# options['flair_name'] = default_config.get('database', 'flair_name')
# options['FLAIR_tags'] = 'Flair'
options['FLAIR_tags'] = str(args.flair_name)


options['roi_tags'] = None

# options['ROI_name'] = default_config.get('database', 'ROI_name')
options['debug'] = True


modalities = [str(options['FLAIR_tags'][0])]
names = ['FLAIR']

options['modalities'] = [n for n, m in
                         zip(names, modalities) if m != 'None']
options['image_tags'] = [m for m in modalities if m != 'None']
options['x_names'] = [n + '_tmp.nii.gz' for n, m in
                      zip(names, modalities) if m != 'None']

options['out_name'] = 'out_seg.nii.gz'
options['skull_stripping'] = False
options['register_modalities'] = False
options['bias_correction'] = False

# preprocessing

if args.register:
    print(CYELLOW + "register modality is activated" + CEND)
    options['register_modalities'] = True
else:
    print(CYELLOW + "register modality is deactivated" + CEND)
    options['register_modalities'] = False

if args.bias:
    print(CYELLOW + "bias correction is activated" + CEND)
    options['bias_correction'] = True
else:
    print(CYELLOW + "bias correction is deactivated" + CEND)
    options['bias_correction'] = False

if args.skull_strip:
    print(CYELLOW + "skull stripping is activated" + CEND)
    options['skull_stripping'] = True
else:
    print(CYELLOW + "skull stripping is deactivated" + CEND)
    options['skull_stripping'] = False
# options['register_modalities_kind'] = (default_config.get('database',
#
#                                                      'register_modalities_Kind'))
options['reg_space'] = 'FlairtoT1'
options['denoise'] = True
options['denoise_iter'] = 3
options['bias_iter'] = 10
options['bias_smooth'] = 20
options['bias_type'] = 1
options['bias_choice'] = 'All'
# options['denoise_iter'] = 3
# options['bias_iter'] = 10
# options['bias_smooth'] = 20
# options['bias_type'] = 1
# options['bias_choice'] = 'All'
if args.reg_space:
    print(CYELLOW + "register modality to space:", str(args.reg_space) + CEND)
    options['reg_space'] = str(args.reg_space)
else:
    # print(CYELLOW + "register modality to space: FlairtoT1" + CEND)
    options['reg_space'] = 'FlairtoT1'



options['batch_prediction'] = False


options['save_tmp'] = True

# net options
# options['gpu_mode'] = default_config.get('model', 'gpu_mode')
options['gpu_number'] = 0
options['pretrained'] = None
options['min_th'] = -0.5
options['fully_convolutional'] = False
options['patch_size'] = (11, 11, 11)
options['weight_paths'] = None
options['train_split'] = 0.25
options['max_epochs'] = 2000
options['patience'] = 50
if args.batch_size:
    options['batch_size'] = int(args.batch_size)
else:
    options['batch_size'] = 128
options['net_verbose'] = 1

options['tensorboard'] = '/path/to/tensorboardlogs'
options['port'] = 8000

# options['load_weights'] = default_config.get('model', 'load_weights')
options['load_weights'] = True
options['randomize_train'] = True

# post processing options
if args.threshold:
    print(CYELLOW + "New  threshold:", str(args.threshold) + CEND)
    options['t_bin'] = float(args.threshold)
else:
    options['t_bin'] = 0.5


options['l_min'] = 10
options['min_error'] = 0.5

# training options  model_1_train
options['full_train'] = True
options['model_1_train'] = True
options['model_2_train'] = True
options['pre_processing'] = False
options['pretrained_model'] = None

options['balanced_training'] = True

options['fract_negative_positive'] = 2.0
options['num_layers'] = None
if args.N4:
     options['use_of_fsl'] = True
else:
     # print(CYELLOW + "N4 run will be proceeded" + CEND)
     options['use_of_fsl'] = False
     if not args.reg_space:
          options['reg_space'] = 'MNI152_T1_1mm.nii.gz'
     else:
          options['reg_space'] = str(args.reg_space)

options['rf'] = False
if args.rf:
    options['rf'] = True
else:
    options['rf'] = False



keys = list(options.keys())
for k in keys:
    value = options[k]
    if value == 'True':
        options[k] = True
    if value == 'False':
        options[k] = False
CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
options['tmp_folder'] = CURRENT_PATH + '/tmp'
options['standard_lib'] = CURRENT_PATH + '/libs/standard'

    # set paths taking into account the host OS
host_os = platform.system()
options['le_path'] = CURRENT_PATH + '/Singularity_le/MNI152_T1_1mm.nii.gz'
if host_os == 'Linux' or 'Darwin':
        options['niftyreg_path'] = CURRENT_PATH + '/libs/linux/niftyreg'
        options['robex_path'] = CURRENT_PATH + '/libs/linux/ROBEX/runROBEX.sh'
        # options['fsl__bet_path'] = CURRENT_PATH + '/libs/linux/fsl/bin'
        # options['fsl__bet_sh'] = CURRENT_PATH + '/libs/linux/fsl/etc/fslconf'

        # options['tensorboard_path'] = CURRENT_PATH + '/libs/bin/tensorboard'
        options['test_slices'] = 256
elif host_os == 'Windows':
        options['niftyreg_path'] = os.path.normpath(
            os.path.join(CURRENT_PATH,
                         'libs',
                         'win',
                         'niftyreg'))

        options['robex_path'] = os.path.normpath(
            os.path.join(CURRENT_PATH,
                         'libs',
                         'win',
                         'ROBEX',
                         'runROBEX.bat'))


        options['test_slices'] = 256
else:
        print("The OS system also here ...", host_os, "is not currently supported.")
        exit()

    # print options when debugging
print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')
print('\x1b[6;30;45m' + 'Train/Test settings' + '\x1b[0m')
print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')
print(" ")
device = str(options['gpu_number'])
print("DEBUG: ", device)
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = device

options['full_train'] = True
options['load_weights'] = True

if args.weights1:

     options['weight_paths1'] = os.path.join(args.weights1)
else:
     options['weight_paths1'] = os.path.join(CURRENT_PATH, 'nets')

if args.weights2:
    options['weight_paths2'] = os.path.join(args.weights2)
else:
     options['weight_paths2'] = os.path.join(CURRENT_PATH, 'nets')

options['net_verbose'] = 0
keys = list(options.keys())
for key in keys:
    print(CRED + key, ':' + CEND, options[key])
print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')
options['task'] = 'testing'
scan_list = os.listdir(options['test_folder'])
scan_list.sort()

print('\x1b[6;30;42m' + '.............................................................' + '\x1b[0m')
print('\x1b[6;30;42m' + 'preprocessing of testing data started....................... ' + '\x1b[0m')
print('\x1b[6;30;42m' + '.............................................................' + '\x1b[0m')
try:
    for scan in scan_list:
        total_time = time.time()
        options['tmp_scan'] = scan
        current_folder = os.path.join(options['test_folder'], scan)
        options['tmp_folder'] = os.path.normpath(
        os.path.join(current_folder, 'tmp'))
        preprocess_scan(current_folder, options)
except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)
print('\x1b[6;30;42m' + '.............................................................' + '\x1b[0m')
print('\x1b[6;30;42m' + 'preprocessing of testing data completed......................' + '\x1b[0m')
print('\x1b[6;30;42m' + '.............................................................' + '\x1b[0m')
keras.backend.set_image_data_format('channels_first')

def transform(Xb, yb):
    """
    handle class for on-the-fly data augmentation on batches.
    Applying 90,180 and 270 degrees rotations and flipping
    """
    # Flip a given percentage of the images at random:
    bs = Xb.shape[0]
    indices = np.random.choice(bs, bs // 2, replace=False)
    x_da = Xb[indices]

    # apply rotation to the input batch
    rotate_90 = x_da[:, :, :, ::-1, :].transpose(0, 1, 2, 4, 3)
    rotate_180 = rotate_90[:, :, :, :: -1, :].transpose(0, 1, 2, 4, 3)
    rotate_270 = rotate_180[:, :, :, :: -1, :].transpose(0, 1, 2, 4, 3)
    # apply flipped versions of rotated patches
    rotate_0_flipped = x_da[:, :, :, :, ::-1]
    rotate_90_flipped = rotate_90[:, :, :, :, ::-1]
    rotate_180_flipped = rotate_180[:, :, :, :, ::-1]
    rotate_270_flipped = rotate_270[:, :, :, :, ::-1]

    augmented_x = np.stack([x_da, rotate_90, rotate_180, rotate_270,
                            rotate_0_flipped,
                            rotate_90_flipped,
                            rotate_180_flipped,
                            rotate_270_flipped],
                            axis=1)

    # select random indices from computed transformations
    r_indices = np.random.randint(0, 3, size=augmented_x.shape[0])

    Xb[indices] = np.stack([augmented_x[i,
                                        r_indices[i], :, :, :, :]
                            for i in range(augmented_x.shape[0])])

    return Xb, yb


def da_generator(x_train, y_train, batch_size=256):
    """
    Keras generator used for training with data augmentation. This generator
    calls the data augmentation function yielding training samples
    """
    num_samples = x_train.shape[0]
    while True:
        for b in range(0, num_samples, batch_size):
            x_ = x_train[b:b+batch_size]
            y_ = y_train[b:b+batch_size]
            x_, y_ = transform(x_, y_)
            yield x_, y_
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def Jaccard_index(y_true, y_pred):
    smooth = 100.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    score = (intersection + smooth) / (union + smooth)
    return score
##################################
def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K] = 1
    # np.random.shuffle(arr)
    return arr


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x



def true_false_positive_loss(y_true, y_pred, p_labels=0, label_smoothing=0, value=0):
    # y_pred_f = tf.reshape(y_pred, [-1])

    # y_pred_f = tf.reshape(y_pred, [-1])
    # this_size = K.int_shape(y_pred_f)[0]
    # # arr_len = this_size
    # # num_ones = 1
    # # arr = np.zeros(arr_len, dtype=int)
    # # if this_size is not None:
    # #      num_ones= np.int32((this_size * value * 100) / 100)
    # #      idx = np.random.choice(range(arr_len), num_ones, replace=False)
    # #      arr[idx] = 1
    # #      p_labels = arr
    # # else:
    # #      p_labels = np.random.randint(2, size=this_size)
    # if this_size is not None:
    #      p_labels = np.random.binomial(1, value, size=this_size)
    #      p_labels = tf.reshape(p_labels, y_pred.get_shape())
    # else:
    #      p_labels =  y_pred

    # p_lprint ('num_classes .....///', num_classes.shape[0])abels = tf.constant(y_pred_f)
    # print ('tf.size(y_pred_f) .....///', tf.size(y_pred_f))
    # num_classes = tf.dtypes.cast((tf.dtypes.cast(tf.size(y_pred_f), tf.int32) *  tf.dtypes.cast(value, tf.int32) * 100) / 100, tf.int32)

    # num_classes = 50
    # print ('num_classes .....', num_classes)
    # p_labels = tf.one_hot(tf.dtypes.cast(tf.zeros_like(y_pred_f), tf.int32), num_classes, 1, 0)
    # p_labels = tf.reduce_max(p_labels, 0)
    # p_labels = tf.reshape(p_labels, tf.shape(y_pred))
    # y_pred = K.constant(y_pred) if not tf.contrib.framework.is_tensor(y_pred) else y_pred
    # y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)

    C11 = tf.math.multiply(y_true, y_pred)
    c_y_pred = 1 - y_pred
    C12 = tf.math.multiply(y_true, c_y_pred)
    weighted_y_pred_u = tf.math.multiply(K.cast(p_labels, y_pred.dtype), y_pred)
    weighted_y_pred_d = 1 - weighted_y_pred_u
    y_pred = tf.math.add(tf.math.multiply(C11, weighted_y_pred_u), tf.math.multiply(C12, weighted_y_pred_d))

    # y_pred /= tf.reduce_sum(y_pred,
    #                             reduction_indices=len(y_pred.get_shape()) - 1,
    #                             keep_dims=True)
    #     # manual computation of crossentropy
    # _EPSILON = 10e-8
    # epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    # y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # loss = - tf.reduce_sum(y_true * tf.log(y_pred),
    #                            reduction_indices=len(y_pred.get_shape()) - 1)

    loss = Jaccard_loss(y_true, y_pred)
    # with tf.GradientTape() as t:
    #     t.watch(y_pred)
    #     dpred = t.gradient(loss, y_pred)

    return loss

    # return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


def false_true_negative_loss(y_true, y_pred, p_labels=0, label_smoothing=0, value=0):
    # arr_len = this_size
    # num_ones = 1
    # arr = np.zeros(arr_len, dtype=int)
    # if this_size is not None:
    #      num_ones= np.int32((this_size * value * 100) / 100)
    #      idx = np.random.choice(range(arr_len), num_ones, replace=False)
    #      arr[idx] = 1
    #      p_labels = arr
    # else:
    #      p_labels = np.random.randint(2, size=this_size)
    # np.random.binomial(1, 0.34, size=10)

    # if this_size is not None:
    #     this_value= np.int32((this_size * value * 100) / 100)
    # else:
    #     this_value =  1

    # p_labels = np.random.randint(2, size=this_size)

    # p_labels = tf.constant(y_pred_f)
    # print ('tf.size(y_pred_f) .....///', tf.size(y_pred_f))
    # num_classes = tf.dtypes.cast((tf.dtypes.cast(tf.size(y_pred_f), tf.int32) *  tf.dtypes.cast(value, tf.int32) * 100) / 100, tf.int32)
    # num_classes = 50
    # print ('num_classes .....', num_classes)
    # p_labels = tf.one_hot(tf.dtypes.cast(tf.zeros_like(y_pred_f), tf.int32), num_classes, 1, 0)
    # # p_labels = tf.reduce_max(p_labels, 0)
    # p_labels = tf.reshape(p_labels, tf.shape(y_pred))
    # y_pred = K.constant(y_pred) if not tf.contrib.framework.is_tensor(y_pred) else y_pred
    # y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)

    c_y_true = 1 - y_true
    c_y_pred = 1 - y_pred
    C21 = tf.math.multiply(c_y_true, y_pred)

    C22 = tf.math.multiply(c_y_true, c_y_pred)
    weighted_y_pred_u = tf.math.multiply(K.cast(p_labels, y_pred.dtype), y_pred)
    weighted_y_pred_d = 1 - weighted_y_pred_u

    y_pred = tf.math.add(tf.math.multiply(C21, weighted_y_pred_u), tf.math.multiply(C22, weighted_y_pred_d))
    # y_pred /= tf.reduce_sum(y_pred,
    #                             reduction_indices=len(y_pred.get_shape()) - 1,
    #                             keep_dims=True)
    #     # manual computation of crossentropy
    # _EPSILON = 10e-8
    # epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    # y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # loss = - tf.reduce_sum(y_true * tf.log(y_pred),
    #                            reduction_indices=len(y_pred.get_shape()) - 1)
    # with tf.GradientTape() as t:
    #     t.watch(y_pred)
    #     dpred = t.gradient(loss, y_pred)
    y_true = 1 - y_true
    loss = Jaccard_loss(y_true, y_pred)

    return loss


def penalty_loss_trace_normalized_confusion_matrix(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred

    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * neg_y_pred)
    sum1 = tp + fn

    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    sum2 = fp + tn

    tp_n = tp / sum1
    fn_n = fn / sum1
    fp_n = fp / sum2
    tn_n = tn / sum2
    trace = (tf.math.square(tp_n) + tf.math.square(tn_n) + tf.math.square(fn_n) + tf.math.square(fp_n))

    with tf.GradientTape() as t:
        t.watch(y_pred)
        pg = t.gradient(trace, y_pred)
    return (1 - trace * 0.5) / 5


def p_loss(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    with tf.GradientTape() as t:
        t.watch(y_pred_f)
        pg = t.gradient(score, y_pred_f)
    return score, pg


def constrain(y_true, y_pred):
    loss, g = p_loss(y_true, y_pred)
    return loss


def constrain_loss(y_true, y_pred):
    return 1 - constrain(y_true, y_pred)


# def augmented_Lagrangian_loss(y_true, y_pred, augment_Lagrangian=1):

#     C_value, pgrad = p_loss(y_true, y_pred)
#     Ld, grad1 = loss_down(y_true, y_pred, from_logits=False, label_smoothing=0, value=C_value)
#     Lu, grad2 = loss_up(y_true, y_pred, from_logits=False, label_smoothing=0, value=C_value)
#     ploss = 1 - C_value
#     # adaptive lagrange multiplier
#     _EPSILON = 10e-8
#     if all(v is not None for v in [grad1, grad2, pgrad]):
#          alm = - ((grad1 + grad2) / pgrad + _EPSILON)
#     else:
#          alm =  augment_Lagrangian
#     ploss = ploss * alm
#     total_loss = Ld + Lu + ploss
#     return total_loss

def calculate_gradient(y_true, y_pred, loss1, loss2):
    with tf.GradientTape(persistent=True) as t:
        t.watch(y_pred)
        g_loss1 = t.gradient(loss1, y_pred)
        g_loss2 = t.gradient(loss2, y_pred)
    # loss, g_constrain = p_loss (y_true, y_pred)
    loss, g_constrain = p_loss(y_true, y_pred)
    return g_loss1, g_loss2, g_constrain


def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return tf.matmul(arg, arg) + arg


def adaptive_lagrange_multiplier(loss1=None, loss2=None, loss3=None):
    _EPSILON = 10e-8
    augment_Lagrangian = 1
    if all(v is not None for v in [loss1, loss2, loss3]):
        res = ((loss1 + loss2) + _EPSILON) / (loss3 + _EPSILON)
        # print ("adaptive_lagrange_multiplier", r)
        return res
    else:
        # print("adaptive_lagrange_multiplier", augment_Lagrangian)
        return augment_Lagrangian


def Individual_loss(y_true, y_pred):
    # C_value = p_loss(y_true, y_pred)
    constrain_l = constrain_loss(y_true, y_pred)
    this_value = (-1 * constrain_l) + 1
    y_pred_f = tf.reshape(y_pred, [-1])
    this_size = K.int_shape(y_pred_f)[0]
    if this_size is not None:
        #  numpy.random.rand(4)
        #  p_labels = np.random.binomial(1, this_value, size=this_size)
        p_labels = rand_bin_array(this_value, this_size)
        p_labels = my_func(np.array(p_labels, dtype=np.float32))
        # p_labels = tf.dtypes.cast((tf.dtypes.cast(tf.size(p_labels), tf.int32)
        p_labels = tf.reshape(p_labels, y_pred.get_shape())

    else:
        p_labels = 0

    loss1 = true_false_positive_loss(y_true, y_pred, p_labels=p_labels, label_smoothing=0, value=this_value)
    loss2 = false_true_negative_loss(y_true, y_pred, p_labels=p_labels, label_smoothing=0, value=this_value)
    grad1, grad2, pgrad = calculate_gradient(y_true, y_pred, loss1, loss2)

    # ploss = 1 - C_value
    # adaptive lagrange multiplier
    # adaptive_lagrange_multiplier(y_true, y_pred):
    # _EPSILON = 10e-8
    # if all(v is not None for v in [grad1, grad2, pgrad]):
    #     return (((grad1 + grad2) + _EPSILON) / pgrad + _EPSILON)
    # else:
    #     return  augment_Lagrangian
    # ploss = ploss * alm
    lm = adaptive_lagrange_multiplier(grad1, grad2, pgrad)
    to_loss = loss1 + loss2 + (lm * constrain_l)
    return to_loss + penalty_loss_trace_normalized_confusion_matrix(
        y_true, y_pred)


##################################



def Symmetric_Hausdorf_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    # Calculating the forward HD: mean(min(each col))
    left = K.maximum(K.minimum(y_true - y_pred, inf), -inf)

    # Calculating the reverse HD: mean(min(each row))
    right = K.maximum(K.minimum(y_pred - y_true, inf), -inf)
    # Calculating mhd
    res = K.maximum(left, right)
    return K.max(res)

def Jaccard_loss(y_true, y_pred):
    loss = 1 - Jaccard_index(y_true, y_pred)
    return loss

def Dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def Combi_Dist_loss(y_true, y_pred):
    y_truec = K.l2_normalize(y_true, axis=-1)
    y_predc = K.l2_normalize(y_pred, axis=-1)

    #loss1 = K.sum(K.abs(y_pred - y_true), axis=-1) + K.mean(K.square(y_pred - y_true), axis=-1)
    #loss2 = -K.mean(y_true_c * y_pred_c, axis=-1) + 100. * K.mean(diff, axis=-1)
    #loss = K.max(loss1+ loss2)
    return K.maximum(K.maximum(K.sum(K.abs(y_pred - y_true), axis=-1) , K.mean(K.square(y_pred - y_true), axis=-1)), -K.sum(y_truec * y_predc, axis=-1))

def accuracy_loss(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    fn = K.sum(y_true * neg_y_pred)
    tp = K.sum(y_true * y_pred)
    acc = (tp + tn) / (tp + tn + fn + fp)
    return 1.0 - acc


def specificity(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    spec = tn / (tn + fp + K.epsilon())
    return spec


def specificity_loss(y_true, y_pred):
        return 1.0 - specificity(y_true, y_pred)


def sensitivity(y_true, y_pred):
    # neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fn = K.sum(y_true * neg_y_pred)
    tp = K.sum(y_true * y_pred)
    sens = tp / (tp + fn + K.epsilon())
    return sens

def sensitivity_loss(y_true, y_pred):
        return 1.0 - sensitivity(y_true, y_pred)

def precision(y_true, y_pred):
    neg_y_true = 1 - y_true
    # neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tp = K.sum(y_true * y_pred)
    pres = tp / (tp + fp + K.epsilon())
    return pres

def precision_loss(y_true, y_pred):
        return 1.0 - precision(y_true, y_pred)

def concatenated_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + Dice_loss(y_true, y_pred) + Jaccard_loss(y_true, y_pred) + \
           Combi_Dist_loss(y_true, y_pred) + Symmetric_Hausdorf_loss(y_true, y_pred) + specificity_loss(y_true, y_pred) + \
           sensitivity_loss(y_true, y_pred) + precision_loss(y_true, y_pred) + accuracy_loss(y_true, y_pred) + Individual_loss(y_true, y_pred)
    #loss = losses.categorical_crossentropy(y_true, y_pred)
    return loss
try:
    print('')
    print('\x1b[6;30;42m' + 'inference started.......................' + '\x1b[0m')
    options['full_train'] = True
    options['load_weights'] = True
    if args.weights1:
        options['weight_paths1'] = os.path.join(args.weights1)
    else:
        options['weight_paths1'] = os.path.join(CURRENT_PATH, 'nets')

    if args.weights2:
        options['weight_paths2'] = os.path.join(args.weights2)
    else:
        options['weight_paths2'] = os.path.join(CURRENT_PATH, 'nets')

    options['net_verbose'] = 0

    model = get_network(options)
    model.compile(loss=concatenated_loss,
                  optimizer='adadelta',
                  metrics=[concatenated_loss, Dice_loss, Jaccard_loss, Combi_Dist_loss,
                           Symmetric_Hausdorf_loss, specificity_loss, sensitivity_loss, precision_loss, accuracy_loss,
                           Individual_loss])
    # if options['debug']:
    #     model.summary()

    # save weights
    net_model_1 = 'model_1'
    if args.weights1:
        net_weights_1 = os.path.join(options['weight_paths1'])
    else:
         net_weights_1 = os.path.join(options['weight_paths1'],
                                 options['experiment'],
                                 'nets', net_model_1 + '.hdf5')

    net1 = {}
    net1['net'] = model
    net1['weights'] = net_weights_1
    net1['history'] = None
    net1['special_name_1'] = net_model_1

    # --------------------------------------------------
    # model 2
    # --------------------------------------------------

    model2 = get_network(options)
    model2.compile(loss=concatenated_loss,
                   optimizer='adadelta',
                   metrics=[concatenated_loss, Dice_loss, Jaccard_loss, Combi_Dist_loss,
                            Symmetric_Hausdorf_loss, specificity_loss, sensitivity_loss, precision_loss, accuracy_loss,
                            Individual_loss])
    # if options['debug']:
    #    model2.summary()

    # save weights
    # save weights
    net_model_2 = 'model_2'
    if args.weights2:
         net_weights_2 = os.path.join(options['weight_paths2'])
    else:
         net_weights_2 = os.path.join(options['weight_paths2'],
                                 options['experiment'],
                                 'nets', net_model_2 + '.hdf5')

    net2 = {}
    net2['net'] = model2
    net2['weights'] = net_weights_2
    net2['history'] = None
    net2['special_name_2'] = net_model_2

    print(net_weights_1)
    print(net_weights_2)

    net1['net'].load_weights(net_weights_1, by_name=True)
    net2['net'].load_weights(net_weights_2, by_name=True)

    model = [net1, net2]

    options['task'] = 'testing'
    scan_list = os.listdir(options['test_folder'])
    scan_list.sort()
    for scan in scan_list:
        total_time = time.time()
        options['tmp_scan'] = scan
        current_folder = os.path.join(options['test_folder'], scan)
        options['tmp_folder'] = os.path.normpath(
            os.path.join(current_folder, 'tmp'))
        seg_time = time.time()

        print("> CNN:", scan, "running WM lesion segmentation")
        sys.stdout.flush()
        options['test_scan'] = scan

        test_x_data = {scan: {m: os.path.join(options['tmp_folder'], n)
                              for m, n in zip(options['modalities'],
                                              options['x_names'])}}

        test_cascaded_model(model, test_x_data, options)

        if options['register_modalities']:
            print(CYELLOW + "Inverting lesion segmentation masks:", CRED + scan + CEND, ".....started!" + CEND)
            invert_registration(current_folder, options)

        print("> INFO:", scan, "CNN Segmentation time: ", round(time.time() - seg_time), "sec")
        print("> INFO:", scan, "total pipeline time: ", round(time.time() - total_time), "sec")

        if options['save_tmp'] is False:
            try:
                os.rmdir(options['tmp_folder'])
                os.rmdir(os.path.join(options['current_folder'],
                                      options['experiment']))
            except:
                pass

    print('\x1b[6;30;42m' + 'inference completed.....................' + '\x1b[0m')
except KeyboardInterrupt:
    print("KeyboardInterrupt has been caught.")
    time.sleep(1)
    os.kill(os.getpid(), signal.SIGTERM)


