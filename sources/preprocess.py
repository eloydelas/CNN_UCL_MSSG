import os
import shutil
import sys
import signal
import subprocess
import time
import platform
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from shutil import which
from medpy.filter.smoothing import anisotropic_diffusion as ans_dif
from nibabel import load as load_nii

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

# def N4():
#     print("N4 bias correction runs.")
#     inputImage = sitk.ReadImage("Flair_reg_brain.nii.gz")
#     # maskImage = sitk.ReadImage("06-t1c_mask.nii.gz")
#     maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
#     sitk.WriteImage(maskImage, "Flair_reg_brain_bias_mask.nii.gz")
#
#     inputImage = sitk.Cast(inputImage,sitk.sitkFloat32)
#
#     corrector = sitk.N4BiasFieldCorrectionImageFilter()
#
#     output = corrector.Execute(inputImage,maskImage)
#     sitk.WriteImage(output,"Flair_reg_brain_bias.nii.gz")
#     print("Finished N4 Bias Field Correction.....")

# if args.fsl == 'yes':
#     print("Fsl (flirt, bet) running ...")
#     logger = logging.getLogger(__name__)
#
#     read_file_path = './Train/'
#     read_folder_list = os.listdir(read_file_path)
#
#     for ii in range(0, len(read_folder_list)):
#         read_folder_name = read_folder_list[ii]
#         print(read_folder_name)
#         folder_path = os.path.join(read_file_path, read_folder_name)
#
#         os.chdir(folder_path)
#         os.system(
#         '/usr/local/fsl/bin/flirt -in Flair.nii -ref ./MNI152_T1_1mm.nii.gz -out Flair_reg.nii.gz -omat Flair_reg.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear')
#
#         # os.system('/usr/local/fsl/bin/bet Flair_reg Flair_reg_brain  -f 0.5 -g 0')
#
#         os.system('python N4()')
#         os.system('rm Flair_reg_brain_bias_mask.nii.gz')



def get_mode(input_data):
    """
    Get the stastical mode
    """
    (_, idx, counts) = np.unique(input_data,
                                 return_index=True,
                                 return_counts=True)
    index = idx[np.argmax(counts)]
    mode = input_data[index]

    return mode


def parse_input_masks(current_folder, options):
    """
    identify input image masks parsing image name labels

    """

    if options['task'] == 'training':
        modalities = options['modalities'][:] + ['lesion']
        image_tags = options['image_tags'][:] + options['roi_tags'][:]
    else:
        modalities = options['modalities'][:]
        image_tags = options['image_tags'][:]

    if options['debug']:
        print("> DEBUG:", "number of input sequences to find:", len(modalities))
    scan = options['tmp_scan']

    print("> PRE:", scan, "identifying input modalities")

    found_modalities = 0

    masks = [m for m in os.listdir(current_folder) if m.find('.nii') > 0]

    for t, m in zip(image_tags, modalities):

        # check first the input modalities
        # find tag

        found_mod = [mask.find(t) if mask.find(t) >= 0
                     else np.Inf for mask in masks]

        if found_mod[np.argmin(found_mod)] is not np.Inf:
            found_modalities += 1
            index = np.argmin(found_mod)
            # generate a new output image modality
            # check for extra dimensions
            input_path = os.path.join(current_folder, masks[index])
            input_sequence = nib.load(input_path)
            input_image = np.squeeze(input_sequence.get_data())
            output_sequence = nib.Nifti1Image(input_image,
                                              affine=input_sequence.affine)
            output_sequence.to_filename(
                os.path.join(options['tmp_folder'], m + '.nii.gz'))

            if options['debug']:
                print("    --> ", masks[index], "as", m, "image")
            masks.remove(masks[index])

    # check that the minimum number of modalities are used
    if found_modalities < len(modalities):
        print("> ERROR:", scan, \
            "does not contain all valid input modalities")
        sys.stdout.flush()
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)


def register_masks(options, extra=None):
    """
    - to doc
    - moving all images to the MPRAGE+192 space

    """

    scan = options['tmp_scan']
    # rigid registration
    os_host = platform.system()
    if os_host == 'Windows':
        reg_exe = 'reg_aladin.exe'
    elif os_host == 'Linux' or 'Darwin':
        reg_exe = 'reg_aladin'
    else:
        print("> ERROR: The OS system", os_host, "is not currently supported.")
    reg_aladin_path=''

    if os_host == 'Windows':
          reg_aladin_path = os.path.join(options['niftyreg_path'], reg_exe)
    elif os_host == 'Linux':
          reg_aladin_path = os.path.join(options['niftyreg_path'], reg_exe)
    elif os_host == 'Darwin':
          reg_aladin_path = os.path.join(options['niftyreg_path'], reg_exe)
    else:
          print('Please install first  NiftyReg in your mac system and try again!')
          sys.stdout.flush()
          time.sleep(1)
          os.kill(os.getpid(), signal.SIGTERM)




    print ('running ....> ',reg_aladin_path)
    if options['reg_space'] == 'FlairtoT1':
        for mod in options['modalities']:
            if mod == 'T1':
                continue

            try:

                if extra is not None:
                    this_path = extra
                    print("> PRE:", scan, "registering", mod, str(extra), " -->space")
                    print(os.path.join(options['tmp_folder'], mod + '.nii.gz'))
                else:
                    this_path = os.path.join(options['tmp_folder'], 'T1.nii.gz')
                    print("> PRE:", scan, "registering", mod, " --> T1 space")
                    print("registering using reg_aladin", mod, " --> T1 space")

                # subprocess.check_output([reg_aladin_path, '-ref', os.path.join(this_path),
                #                          '-flo', os.path.join(options['tmp_folder'], mod + '.nii.gz'),
                #                          '-aff', os.path.join(options['tmp_folder'], mod + '_transf.txt'),
                #                          '-res', os.path.join(options['tmp_folder'], 'r' + mod + '.nii.gz')])
                # bash = 'bash'
                # arg1 =str(reg_aladin_path) + '  ' + '-ref' + '  ' + str(this_path) + '  ' + '-flo' +\
                #        '  ' + str(os.path.join(options['tmp_folder'], mod + '.nii.gz')) + '  ' + '-aff' + '  ' +\
                # str(os.path.join(options['tmp_folder'], mod + '_transf.txt')) + '  ' + '-res' + '  ' + str(os.path.join(options['tmp_folder'], 'r' + mod + '.nii.gz'))
                # print(arg1)
                # os.system(arg1)

                process = subprocess.Popen([reg_aladin_path, '-ref', os.path.join(this_path),
                                          '-flo', os.path.join(options['tmp_folder'], mod + '.nii.gz'),
                                          '-aff', os.path.join(options['tmp_folder'], mod + '_transf.txt'),
                                          '-res', os.path.join(options['tmp_folder'], 'r' + mod + '.nii.gz')],
                                           stdout=subprocess.PIPE,
                                           universal_newlines=True)

                while True:
                    output = process.stdout.readline()
                    print(output.strip())
                    # Do something else
                    return_code = process.poll()
                    if return_code is not None:
                        print('RETURN CODE', return_code)
                        # Process has finished, read rest of the output
                        for output in process.stdout.readlines():
                            print(output.strip())
                        break
            except:
                print("> ERROR:", scan, "registering masks on  ", mod, "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

    # if training, the lesion mask is also registered through the T1 space.
    # Assuming that the refefence lesion space was FLAIR.
    if options['reg_space'] == 'FlairtoT1':
        if options['task'] == 'training':
            # rigid registration
            os_host = platform.system()
            if os_host == 'Windows':
                reg_exe = 'reg_resample.exe'
            elif os_host == 'Linux' or 'Darwin':
                reg_exe = 'reg_resample'
            else:
                print("> ERROR: The OS system", os_host, "is not currently supported.")

            reg_resample_path = ''

            if os_host == 'Windows':
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_exe)
            elif os_host == 'Linux':
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_exe)
            elif os_host == 'Darwin':
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_exe)
            else:
                print('Please install first  NiftyReg in your mac system and try again!')
                sys.stdout.flush()
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)
            print('running ....> ', reg_resample_path)


            try:

                if extra is not None:
                    this_path = extra
                    print("> PRE:", scan, "registering", mod, str(extra), " -->space")
                else:
                    this_path = os.path.join(options['tmp_folder'], 'T1.nii.gz')
                    print("> PRE:", scan, "resampling the lesion mask --> T1 space")
                    print("registering using reg_resample", mod, " --> T1 space")

                # subprocess.check_output([reg_resample_path, '-ref',this_path,
                #                          '-flo', os.path.join(options['tmp_folder'], 'lesion'),
                #                          '-trans', os.path.join(options['tmp_folder'], 'FLAIR_transf.txt'),
                #                          '-res', os.path.join(options['tmp_folder'], 'lesion.nii.gz'),
                #                          '-inter', '0'])
                process = subprocess.Popen([reg_resample_path, '-ref',this_path,
                                         '-flo', os.path.join(options['tmp_folder'], 'lesion'),
                                         '-trans', os.path.join(options['tmp_folder'], 'FLAIR_transf.txt'),
                                         '-res', os.path.join(options['tmp_folder'], 'lesion.nii.gz'),
                                         '-inter', '0'],
                                           stdout=subprocess.PIPE,
                                           universal_newlines=True)

                while True:
                    output = process.stdout.readline()
                    print(output.strip())
                    # Do something else
                    return_code = process.poll()
                    if return_code is not None:
                        print('RETURN CODE', return_code)
                        # Process has finished, read rest of the output
                        for output in process.stdout.readlines():
                            print(output.strip())
                        break
            except:
                print("> ERROR:", scan, "registering masks on  ", mod, "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

    if options['reg_space'] == 'T1toFlair':
        for mod in options['modalities']:
            if mod == 'FLAIR':
                continue

            try:

                if extra is not None:
                    this_path = extra
                    print("> PRE:", scan, "registering", mod, str(extra), " -->space")
                else:
                    this_path = os.path.join(options['tmp_folder'], 'FLAIR.nii.gz')
                    print("> PRE:", scan, "registering", mod, " --> Flair space")
                    flirt = 'flirt'
                    print("registering using reg_aladin", mod, " --> Flair space")
                # subprocess.check_output([reg_aladin_path, '-ref',this_path,
                #                          '-flo', os.path.join(options['tmp_folder'], mod + '.nii.gz'),
                #                          '-aff', os.path.join(options['tmp_folder'], mod + '_transf.txt'),
                #                          '-res', os.path.join(options['tmp_folder'], 'r' + mod + '.nii.gz')])
                process = subprocess.Popen([reg_aladin_path, '-ref',this_path,
                                         '-flo', os.path.join(options['tmp_folder'], mod + '.nii.gz'),
                                         '-aff', os.path.join(options['tmp_folder'], mod + '_transf.txt'),
                                         '-res', os.path.join(options['tmp_folder'], 'r' + mod + '.nii.gz')],
                                           stdout=subprocess.PIPE,
                                           universal_newlines=True)

                while True:
                    output = process.stdout.readline()
                    print(output.strip())
                    # Do something else
                    return_code = process.poll()
                    if return_code is not None:
                        print('RETURN CODE', return_code)
                        # Process has finished, read rest of the output
                        for output in process.stdout.readlines():
                            print(output.strip())
                        break
            except:
                print("> ERROR:", scan, "registering masks on  ", mod, "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

        # if training, the lesion mask is also registered through the T1 space.
        # Assuming that the refefence lesion space was FLAIR.
    if options['reg_space'] == 'T1toFlair':
        if options['task'] == 'training':
            # rigid registration
            os_host = platform.system()
            if os_host == 'Windows':
                reg_exe = 'reg_resample.exe'
            elif os_host == 'Linux' or 'Darwin':
                reg_exe = 'reg_resample'
            else:
                print("> ERROR: The OS system", os_host, "is not currently supported.")

            reg_resample_path = ''

            if os_host == 'Windows':
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_exe)
            elif os_host == 'Linux':
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_exe)
            elif os_host == 'Darwin':
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_exe)
            else:
                print('Please install first  NiftyReg in your mac system and try again!')
                sys.stdout.flush()
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)
            print('running ....> ', reg_resample_path)

            try:
                if extra is not None:
                    this_path = extra
                    print("> PRE:", scan, "registering", mod, str(extra), " -->space")
                else:
                    this_path = os.path.join(options['tmp_folder'], 'FLAIR.nii.gz')
                    print("> PRE:", scan, "resampling the lesion mask --> Flair space")
                    flirt = 'flirt'
                    print("registering using reg_resample", mod, " --> Flair space")
                # subprocess.check_output([reg_resample_path, '-ref',
                #                          this_path,
                #                          '-flo', os.path.join(options['tmp_folder'], 'lesion'),
                #                          '-trans', os.path.join(options['tmp_folder'], 'T1_transf.txt'),
                #                          '-res', os.path.join(options['tmp_folder'], 'lesion.nii.gz'),
                #                          '-inter', '0'])
                process = subprocess.Popen([reg_resample_path, '-ref',
                                         this_path,
                                         '-flo', os.path.join(options['tmp_folder'], 'lesion'),
                                         '-trans', os.path.join(options['tmp_folder'], 'T1_transf.txt'),
                                         '-res', os.path.join(options['tmp_folder'], 'lesion.nii.gz'),
                                         '-inter', '0'],
                                           stdout=subprocess.PIPE,
                                           universal_newlines=True)

                while True:
                    output = process.stdout.readline()
                    print(output.strip())
                    # Do something else
                    return_code = process.poll()
                    if return_code is not None:
                        print('RETURN CODE', return_code)
                        # Process has finished, read rest of the output
                        for output in process.stdout.readlines():
                            print(output.strip())
                        break
            except:
                print("> ERROR:", scan, "registering masks on  ", mod, "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

    if  options['reg_space'] != 'FlairtoT1' and  options['reg_space'] != 'T1toFlair':
        print("registration to standard space:", options['reg_space'])
        print('using ....> ', options['standard_lib'])
        for mod in options['modalities']:

            try:
                print("> PRE:", scan, "registering", mod, "--->",  options['reg_space'])
                flirt = 'flirt'
                print("registering using reg_aladin", mod, options['reg_space'])
                # subprocess.check_output(['reg_aladin', '-ref',
                #                          os.path.join(options['standard_lib'], options['reg_space']),
                #                          '-flo', os.path.join(options['tmp_folder'], mod + '.nii.gz'),
                #                          '-aff', os.path.join(options['tmp_folder'], mod + '_transf.txt'),
                #                          '-res', os.path.join(options['tmp_folder'], 'r' + mod + '.nii.gz')])
                process = subprocess.Popen(['reg_aladin', '-ref',
                                         os.path.join(options['standard_lib'], options['reg_space']),
                                         '-flo', os.path.join(options['tmp_folder'], mod + '.nii.gz'),
                                         '-aff', os.path.join(options['tmp_folder'], mod + '_transf.txt'),
                                         '-res', os.path.join(options['tmp_folder'], 'r' + mod + '.nii.gz')],
                                           stdout=subprocess.PIPE,
                                           universal_newlines=True)

                while True:
                    output = process.stdout.readline()
                    print(output.strip())
                    # Do something else
                    return_code = process.poll()
                    if return_code is not None:
                        print('RETURN CODE', return_code)
                        # Process has finished, read rest of the output
                        for output in process.stdout.readlines():
                            print(output.strip())
                        break
            except:
                print("> ERROR:", scan, "registering masks on  ", mod, "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

    if  options['reg_space'] != 'FlairtoT1' and  options['reg_space'] != 'T1toFlair':
        print("resampling the lesion mask ----->:", options['reg_space'])            
        if options['task'] == 'training':
            # rigid registration
            os_host = platform.system()
        if os_host == 'Windows':
            reg_exe = 'reg_resample.exe'
        elif os_host == 'Linux' or 'Darwin':
            reg_exe = 'reg_resample'
        else:
            print("> ERROR: The OS system", os_host, "is not currently supported.")

        reg_resample_path = ''

        if os_host == 'Windows':
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_exe)
        elif os_host == 'Linux':
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_exe)
        elif os_host == 'Darwin':
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_exe)
        else:
            print('Please install first  NiftyReg in your mac system and try again!')
            sys.stdout.flush()
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

        print('running ....> ', reg_resample_path)

        try:
            print("> PRE:", scan, "resampling the lesion mask -->",options['reg_space'])
            flirt = 'flirt'
            print("registering using reg_resample", mod, options['reg_space'])
            # subprocess.check_output(['reg_resample', '-ref',
            #                              os.path.join(options['standard_lib'], options['reg_space']),
            #                              '-flo', os.path.join(options['tmp_folder'], 'lesion'),
            #                              '-trans', os.path.join(options['tmp_folder'], 'FLAIR_transf.txt'),
            #                              '-res', os.path.join(options['tmp_folder'], 'lesion.nii.gz'),
            #                              '-inter', '0'])
            process = subprocess.Popen(['reg_resample', '-ref',
                                         os.path.join(options['standard_lib'], options['reg_space']),
                                         '-flo', os.path.join(options['tmp_folder'], 'lesion'),
                                         '-trans', os.path.join(options['tmp_folder'], 'FLAIR_transf.txt'),
                                         '-res', os.path.join(options['tmp_folder'], 'lesion.nii.gz'),
                                         '-inter', '0'],
                                       stdout=subprocess.PIPE,
                                       universal_newlines=True)

            while True:
                output = process.stdout.readline()
                print(output.strip())
                # Do something else
                return_code = process.poll()
                if return_code is not None:
                    print('RETURN CODE', return_code)
                    # Process has finished, read rest of the output
                    for output in process.stdout.readlines():
                        print(output.strip())
                    break
        except:
            print("> ERROR:", scan, "registering masks on  ", mod, "quiting program.")
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

def bias_correction(options):
    """
    Bias correction of  masks [if large differences, bias correction is needed!]
    Using FSL (https://fsl.fmrib.ox.ac.uk/)

    """
    scan = options['tmp_scan']
    if options['task'] == 'training':
         current_folder = os.path.join(options['train_folder'], scan)
         options['bias_folder'] = os.path.normpath(os.path.join(current_folder,'bias'))
    else:
        current_folder = os.path.join(options['test_folder'], scan)
        options['bias_folder'] = os.path.normpath(os.path.join(current_folder,'bias'))    
    try:
        # os.rmdir(os.path.join(current_folder,  'tmp'))
        if options['task'] == 'training':
           os.mkdir(options['bias_folder'])
           print ("bias folder is created for training!")
        else: 
           os.mkdir(options['bias_folder'])
           print ("bias folder is created for testing!")  
    except:
        if os.path.exists(options['bias_folder']) is False:
            print("> ERROR:",  scan, "I can not create bias folder for", current_folder, "Quiting program.")

        else:
            pass

                                                              
   
    # os_host = platform.system()
    print('please be sure FSL is installed in your system, or install FSL in your system and try again!')
    print('\x1b[6;30;42m' + 'Note that the Bias Correction in general can take a long time to finish!' + '\x1b[0m') 
    it =str(options['bias_iter'])
    smooth = str(options['bias_smooth'])  
    type = str(options['bias_type']) 
    
  
    if options['bias_choice'] == 'All':
        BIAS = options['modalities']
    if options['bias_choice'] == 'FLAIR':
        BIAS = ['FLAIR']
    if options['bias_choice'] == 'T1':
        BIAS = ['T1']
    if options['bias_choice'] == 'MOD3':
        BIAS = ['MOD3']  
    if options['bias_choice'] == 'MOD4':
        BIAS = ['MOD4']              


    for mod in BIAS:

        # current_image = mod + '.nii.gz' if mod == 'T1'\  current_image = mod
        try:
            if not options['use_of_fsl']:
                if options['debug']:
                   print("> DEBUG: Bias correction ......> ", mod)
                print("> PRE:", scan, "Bias correction of", mod, "------------------------------->")
                input_scan = mod + '.nii.gz' 
            
                shutil.copy(os.path.join(options['tmp_folder'],
                                         input_scan),
                            os.path.join(options['bias_folder'],
                                         input_scan))
                                        
                fslm = 'fslmaths'
                ft = 'fast'
                fslsf = 'fslsmoothfill'
                output = subprocess.check_output([fslm, os.path.join(options['bias_folder'], input_scan),
                                         '-mul', '0', os.path.join(options['bias_folder'], mod+'lesionmask.nii.gz')], stderr=subprocess.STDOUT)
                subprocess.check_output([fslm, os.path.join(options['bias_folder'], mod+'lesionmask.nii.gz'),
                                         '-bin', os.path.join(options['bias_folder'], mod+'lesionmask.nii.gz')])
                subprocess.check_output([fslm, os.path.join(options['bias_folder'], mod+'lesionmask.nii.gz'),
                                         '-binv', os.path.join(options['bias_folder'], mod+'lesionmaskinv.nii.gz')])
                 
                print(CYELLOW + "Bias correction of", CRED + mod  + CEND , "(step one is done!)" + CEND)                                                         


                subprocess.check_output([fslm, os.path.join(options['bias_folder'], input_scan),
                                         os.path.join(options['bias_folder'], mod + '_initfast2_brain.nii.gz')])
                subprocess.check_output([fslm, os.path.join(options['bias_folder'], mod + '_initfast2_brain.nii.gz'), '-bin', 
                                         os.path.join(options['bias_folder'], mod + '_initfast2_brain_mask.nii.gz')])
                subprocess.check_output([fslm, os.path.join(options['bias_folder'], mod + '_initfast2_brain.nii.gz'), 
                                         os.path.join(options['bias_folder'], mod + '_initfast2_restore.nii.gz')]) 
                subprocess.check_output([fslm, os.path.join(options['bias_folder'], mod + '_initfast2_restore.nii.gz'), '-mas', 
                                         os.path.join(options['bias_folder'], mod+'lesionmaskinv.nii.gz'), 
                                         os.path.join(options['bias_folder'], mod + '_initfast2_maskedrestore.nii.gz')]) 

                print(CYELLOW + "Bias correction of", CRED + mod  + CEND , "(step two is done!)" + CEND) 


                # subprocess.check_output([ft, '-o', os.path.join(options['bias_folder'], mod+'_fast'), '-l', '20', '-b', '-B', 
                #                          '-t', '1', '--iter=10', '--nopve', '--fixed=0', '-v', 
                #                          os.path.join(options['bias_folder'], mod + '_initfast2_maskedrestore.nii.gz')])

                subprocess.check_output([ft, '-o', os.path.join(options['bias_folder'], mod+'_fast'), '-l', smooth, '-b', '-B', 
                                         '-t', type , it , '--nopve', '--fixed=0', '-v', 
                                         os.path.join(options['bias_folder'], mod + '_initfast2_maskedrestore.nii.gz')])

                subprocess.check_output([fslm, os.path.join(options['bias_folder'], input_scan), '-div',
                                         os.path.join(options['bias_folder'], mod + '_fast_restore.nii.gz'), '-mas',
                                         os.path.join(options['bias_folder'], mod + '_initfast2_brain_mask.nii.gz'),
                                         os.path.join(options['bias_folder'], mod + '_fast_totbias.nii.gz')])
                subprocess.check_output([fslm, os.path.join(options['bias_folder'], mod + '_initfast2_brain_mask.nii.gz'), 
                                        '-ero', '-ero', '-ero', '-ero', '-mas', 
                                        os.path.join(options['bias_folder'], mod+'lesionmaskinv.nii.gz'),
                                        os.path.join(options['bias_folder'], mod + '_initfast2_brain_mask2.nii.gz')]) 
                subprocess.check_output([fslm, os.path.join(options['bias_folder'], mod + '_fast_totbias.nii.gz'), '-sub', '1',
                                        os.path.join(options['bias_folder'], mod + '_fast_totbias.nii.gz')]) 


                print(CYELLOW + "Bias correction of", CRED + mod  + CEND , "(step three is done!)" + CEND)



                subprocess.check_output([fslsf, '-i', os.path.join(options['bias_folder'], mod + '_fast_totbias.nii.gz'), '-m',
                                        os.path.join(options['bias_folder'], mod + '_initfast2_brain_mask2.nii.gz'),'-o',
                                        os.path.join(options['bias_folder'], mod + '_fast_bias.nii.gz')]) 
                subprocess.check_output([fslm, os.path.join(options['bias_folder'], mod + '_fast_bias.nii.gz'),'-add', '1',
                                        os.path.join(options['bias_folder'], mod + '_fast_bias.nii.gz')])  
                subprocess.check_output([fslm, os.path.join(options['bias_folder'], mod + '_fast_totbias.nii.gz'),'-add', '1',
                                        os.path.join(options['bias_folder'], mod + '_fast_totbias.nii.gz')])  
                subprocess.check_output([fslm, os.path.join(options['bias_folder'], input_scan),'-div', 
                                        os.path.join(options['bias_folder'], mod + '_fast_bias.nii.gz'),     
                                        os.path.join(options['bias_folder'], mod + '_biascorr.nii.gz')])
                subprocess.check_output([fslm, os.path.join(options['bias_folder'], input_scan),'-div', 
                                        os.path.join(options['bias_folder'], mod + '_fast_bias.nii.gz'),     
                                        os.path.join(options['bias_folder'], mod + '_biascorr.nii.gz')])
                print(CYELLOW + "Replacing the", CRED + mod  + CEND, CGREEN+ "with a new bias corrected version of it in tmp folder" + CEND)                         

                shutil.copy(os.path.join(options['bias_folder'], mod + '_biascorr.nii.gz'),
                            os.path.join(options['tmp_folder'], mod + '.nii.gz'))
                # shutil.copy(os.path.join(options['bias_folder'], mod + '_biascorr.nii.gz'),
                #             os.path.join(options['tmp_folder'], 'bc' + mod + '.nii.gz'))


                print(CYELLOW + "Bias correction of", CRED + mod  + CEND , "(is completed!)" + CEND)
            else:
                 pass


         
                                             
        except:
                
                # print("err: '{}'".format(output))
                print("> ERROR:", scan, "Bias correction of  ", mod,  "quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)


def denoise_masks(options):
    """
    Denoise input masks to reduce noise.
    Using anisotropic Diffusion (Perona and Malik)

    """
    # if options['register_modalities_kind'] != 'FlairtoT1' and  options['register_modalities_kind'] != 'T1toFlair':
    #     print("registration must be either FlairtoT1 or T1toFlair and not", options['register_modalities_kind'])
    #     print("> ERROR:", "quiting program.")
    #     sys.stdout.flush()
    #     time.sleep(1)
    #     os.kill(os.getpid(), signal.SIGTERM)

    for mod in options['modalities']:

        # current_image = mod + '.nii.gz' if mod == 'T1'\
        #                 else 'r' + mod + '.nii.gz'

        if options['reg_space'] == 'T1toFlair':
            current_image = mod + '.nii.gz' if mod == 'FLAIR' \
                else 'r' + mod + '.nii.gz'

        if options['reg_space'] == 'FlairtoT1':
            current_image = mod + '.nii.gz' if mod == 'T1' \
                else 'r' + mod + '.nii.gz'
        if  options['reg_space'] != 'FlairtoT1' and  options['reg_space'] != 'T1toFlair':
            current_image ='r' + mod + '.nii.gz'

        tmp_scan = nib.load(os.path.join(options['tmp_folder'],
                                         current_image))

        tmp_scan.get_data()[:] = ans_dif(tmp_scan.get_data(),
                                         niter=options['denoise_iter'])

        tmp_scan.to_filename(os.path.join(options['tmp_folder'],
                                          'd' + current_image))
        if options['debug']:
            print("> DEBUG: Denoising ", current_image)


def skull_strip(options):

    os_host=platform.system()
    scan = options['tmp_scan']
    if options['reg_space'] == 'FlairtoT1':

            t1_im = os.path.join(options['tmp_folder'], 'drFLAIR.nii.gz')
            t1_st_im = os.path.join(options['tmp_folder'], 'FLAIR_tmp.nii.gz')

            try:
                print("> PRE:", scan, "skull_stripping the Flair modality")
                if os_host == 'Windows':
                    print("skull_stripping the Flair modality on",os_host, "system")
                    bet = 'bet'
                    # if options['use_of_fsl']:
                    #     subprocess.check_output([bet,
                    #                          t1_im,
                    #                          t1_st_im, '-f', '0.5', '-g', '0'])
                    # else:
                    subprocess.check_output([options['robex_path'],
                                             t1_im,
                                             t1_st_im])
                elif os_host == 'Linux':
                    print("skull_stripping the Flair modality on ",os_host, "system")
                    bet = 'bet'
                    # if options['use_of_fsl']:
                    #     subprocess.check_output([bet,
                    #                          t1_im,
                    #                          t1_st_im, '-f', '0.5', '-g', '0'])
                    # else:
                    out = subprocess.check_output(["bash", options['robex_path'],
                                              t1_im,
                                              t1_st_im])

                elif os_host == 'Darwin':
                    print("skull_stripping the Flair modality on",os_host, "system")
                    bet = 'bet'
                    # if options['use_of_fsl']:
                    #     subprocess.check_output([bet,
                    #                          t1_im,
                    #                          t1_st_im, '-f', '0.5', '-g', '0'])
                    # else:
                    subprocess.check_output([bet,
                                             t1_im,
                                             t1_st_im, '-R', '-S', '-B'])
                    # os.system('/usr/local/fsl/bin/bet Flair_reg Flair_reg_brain  -f 0.5 -g 0')
                else:
                    print('Please install first  FSL in your mac system and try again!')
                    sys.stdout.flush()
                    time.sleep(1)
                    os.kill(os.getpid(), signal.SIGTERM)

            except:
                print("> ERROR:", scan, "registering masks, quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

            brainmask = nib.load(t1_st_im).get_data() > 1
            for mod in options['modalities']:

                if mod == 'FLAIR':
                    continue

                # apply the same mask to the rest of modalities to reduce
                # computational time

                print('> PRE: ', scan, 'Applying skull mask to ', mod, 'image')
                current_mask = os.path.join(options['tmp_folder'],
                                            'dr' + mod + '.nii.gz')
                current_st_mask = os.path.join(options['tmp_folder'],
                                               mod + '_tmp.nii.gz')

                mask = nib.load(current_mask)
                mask_nii = mask.get_data()
                mask_nii[brainmask == 0] = 0
                mask.get_data()[:] = mask_nii
                mask.to_filename(current_st_mask)



    if options['reg_space'] == 'T1toFlair':


            # t1_im = os.path.join(options['tmp_folder'], 'dT1.nii.gz')
            # t1_st_im = os.path.join(options['tmp_folder'], 'T1_tmp.nii.gz')
            t1_im = os.path.join(options['tmp_folder'], 'drFLAIR.nii.gz')
            t1_st_im = os.path.join(options['tmp_folder'], 'FLAIR_tmp.nii.gz')

            try:
                print("> PRE:", scan, "skull_stripping the Flair modality")
                if os_host == 'Windows':
                    print("skull_stripping the Flair modality on", os_host, "system")
                    bet = 'bet'
                    # if options['use_of_fsl']:
                    #     subprocess.check_output([bet,
                    #                          t1_im,
                    #                          t1_st_im, '-f', '0.5', '-g', '0'])
                    # else:

                    subprocess.check_output([options['robex_path'],
                                             t1_im,
                                             t1_st_im])
                elif os_host == 'Linux':
                    print("skull_stripping the Flair modality on ",os_host, "system")
                    bet = 'bet'
                    # if options['use_of_fsl']:
                    #     subprocess.check_output([bet,
                    #                          t1_im,
                    #                          t1_st_im, '-f', '0.5', '-g', '0'])
                    # else:
                    out = subprocess.check_output(["bash", options['robex_path'],
                                              t1_im,
                                              t1_st_im])
                    # print(options['robex_path'])
                    # p = subprocess.Popen([options['robex_path'],
                    #                          t1_im,
                    #                          t1_st_im], stdout=subprocess.PIPE)

                    # print(p.communicate())

                elif os_host == 'Darwin':
                    print("skull_stripping the Flair modality on",os_host, "system")
                    bet = 'bet'
                    # if options['use_of_fsl']:
                    #     subprocess.check_output([bet,
                    #                              t1_im,
                    #                              t1_st_im, '-f', '0.5', '-g', '0'])
                    # else:
                    subprocess.check_output([bet,
                                             t1_im,
                                             t1_st_im, '-R', '-S', '-B'])
                else:
                    print('Please install first  FSL in your mac system and try again!')
                    sys.stdout.flush()
                    time.sleep(1)
                    os.kill(os.getpid(), signal.SIGTERM)

            except:
                print("> ERROR:", scan, "registering masks, quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

            brainmask = nib.load(t1_st_im).get_data() > 1
            for mod in options['modalities']:

                if mod == 'FLAIR':
                    continue

                # apply the same mask to the rest of modalities to reduce
                # computational time

                print('> PRE: ', scan, 'Applying skull mask to ', mod, 'image')
                current_mask = os.path.join(options['tmp_folder'],
                                            'dr' + mod + '.nii.gz')
                current_st_mask = os.path.join(options['tmp_folder'],
                                               mod + '_tmp.nii.gz')

                mask = nib.load(current_mask)
                mask_nii = mask.get_data()
                mask_nii[brainmask == 0] = 0
                mask.get_data()[:] = mask_nii
                mask.to_filename(current_st_mask)

    if  options['reg_space'] != 'FlairtoT1' and  options['reg_space'] != 'T1toFlair':    

            # t1_im = os.path.join(options['tmp_folder'], 'dT1.nii.gz')
            # t1_st_im = os.path.join(options['tmp_folder'], 'T1_tmp.nii.gz')
            t1_im = os.path.join(options['tmp_folder'], 'drFLAIR.nii.gz')
            t1_st_im = os.path.join(options['tmp_folder'], 'FLAIR_tmp.nii.gz')

            try:
                print("> PRE:", scan, "skull_stripping the Flair modality")
                if os_host == 'Windows':
                    print("skull_stripping the Flair modality on",os_host, "system")
                    bet = 'bet'
                    # if options['use_of_fsl']:
                    #     subprocess.check_output([bet,
                    #                          t1_im,
                    #                          t1_st_im, '-f', '0.5', '-g', '0'])
                    # else:
                    subprocess.check_output([options['robex_path'],
                                             t1_im,
                                             t1_st_im])
                elif os_host == 'Linux':
                    print("skull_stripping the Flair modality on ",os_host, "system")
                    bet = 'bet'
                    # if options['use_of_fsl']:
                    #     subprocess.check_output([bet,
                    #                          t1_im,
                    #                          t1_st_im, '-f', '0.5', '-g', '0'])
                    # else:
                    out = subprocess.check_output(["bash", options['robex_path'],
                                              t1_im,
                                              t1_st_im])
                    # print(options['robex_path'])
                    # p = subprocess.Popen([options['robex_path'],
                    #                          t1_im,
                    #                          t1_st_im], stdout=subprocess.PIPE)

                    # print(p.communicate())

                elif os_host == 'Darwin':
                    print("skull_stripping the Flair modality on",os_host, "system")
                    bet = 'bet'
                    # if options['use_of_fsl']:
                    #     subprocess.check_output([bet,
                    #                              t1_im,
                    #                              t1_st_im, '-f', '0.5', '-g', '0'])
                    # else:
                    subprocess.check_output([bet,
                                             t1_im,
                                             t1_st_im, '-R', '-S', '-B'])
                else:
                    print('Please install first  FSL in your mac system and try again!')
                    sys.stdout.flush()
                    time.sleep(1)
                    os.kill(os.getpid(), signal.SIGTERM)

            except:
                print("> ERROR:", scan, "registering masks, quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

            brainmask = nib.load(t1_st_im).get_data() > 1
            for mod in options['modalities']:

                if mod == 'FLAIR':
                    continue

                # apply the same mask to the rest of modalities to reduce
                # computational time

                print('> PRE: ', scan, 'Applying skull mask to ', mod, 'image')
                current_mask = os.path.join(options['tmp_folder'],
                                            'dr' + mod + '.nii.gz')
                current_st_mask = os.path.join(options['tmp_folder'],
                                               mod + '_tmp.nii.gz')

                mask = nib.load(current_mask)
                mask_nii = mask.get_data()
                mask_nii[brainmask == 0] = 0
                mask.get_data()[:] = mask_nii
                mask.to_filename(current_st_mask)

            image = load_nii(os.path.join(options['tmp_folder'], 'FLAIR_tmp.nii.gz'))
            image_norm = zscore_normalize(image, mask=None)
            image_norm.to_filename(os.path.join(options['tmp_folder'], 'normalized_' + '.nii.gz'))

def zscore_normalize(img, mask=None):
    """
    normalize a target image by subtracting the mean of the whole brain
    and dividing by the standard deviation
    Args:
        img (nibabel.nifti1.Nifti1Image): target MR brain image
        mask (nibabel.nifti1.Nifti1Image): brain mask for img
    Returns:
        normalized (nibabel.nifti1.Nifti1Image): img with WM mean at norm_value
    """

    img_data = img.get_data()
    if mask is not None and not isinstance(mask, str):
        mask_data = mask.get_data()
    elif mask == 'nomask':
        mask_data = img_data == img_data
    else:
        mask_data = img_data > img_data.mean()
    logical_mask = mask_data == 1  # force the mask to be logical type
    mean = img_data[logical_mask].mean()
    std = img_data[logical_mask].std()
    normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
    return normalized

def preprocess_scan(current_folder, options):

    preprocess_time = time.time()

    scan = options['tmp_scan']
    try:
        # os.rmdir(os.path.join(current_folder,  'tmp'))
        os.mkdir(options['tmp_folder'])
    except:
        if os.path.exists(options['tmp_folder']) is False:
            print("> ERROR:",  scan, "I can not create tmp folder for", current_folder, "Quiting program.")

        else:
            pass

    # --------------------------------------------------
    # find modalities
    # --------------------------------------------------
    id_time = time.time()
    parse_input_masks(current_folder, options)
    print("> INFO:", scan, "elapsed time: ", round(time.time() - id_time), "sec")

    print(CYELLOW + "parse of current input mask done!" + CEND)

    if options['use_of_fsl'] is True:
            print(CYELLOW + "preprocessing using N4 algorithm (N4BiasFieldCorrection)!" + CEND)

            print('register modalities!')
            # os.system(
            #     '/usr/local/fsl/bin/flirt -in Flair.nii -ref ./MNI152_T1_1mm.nii.gz -out Flair_reg.nii.gz -omat Flair_reg.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear') first
            # print(CBLUE2 + "Registration started... moving all images to the MPRAGE+192 space" + CEND)
            # reg_time = time.time()
            # print('Path to Le data:', options['le_path'])
            # # register_masks(options, extra=options['le_path'])
            # # bash = 'bash'
            # os_host = platform.system()
            # reg_exe = ''
            # if os_host == 'Windows':
            #     reg_exe = 'reg_aladin.exe'
            # elif os_host == 'Linux' or 'Darwin':
            #     reg_exe = 'reg_aladin'
            # else:
            #     print("> ERROR: The OS system", os_host, "is not currently supported.")
            # reg_aladin_path = ''
            #
            # if os_host == 'Windows':
            #     reg_aladin_path = os.path.join(options['niftyreg_path'], reg_exe)
            # elif os_host == 'Linux':
            #     reg_aladin_path = os.path.join(options['niftyreg_path'], reg_exe)
            # elif os_host == 'Darwin':
            #     reg_aladin_path = os.path.join(options['niftyreg_path'], reg_exe)
            # else:
            #     print('Please install first  NiftyReg in your mac system and try again!')
            #     sys.stdout.flush()
            #     time.sleep(1)
            #     os.kill(os.getpid(), signal.SIGTERM)
            # arg1 ='bash' + '  ' +  str(reg_aladin_path) + '  ' + '-ref' + '  ' + str(options['le_path']) + '  ' + '-flo' +\
            #        '  ' + str(os.path.join(options['tmp_folder'], 'FLAIR.nii.gz')) + '  ' + '-aff' + '  ' +\
            # str(os.path.join(options['tmp_folder'], 'FLAIR_transf.txt')) + '  ' + '-res' + '  ' + str(os.path.join(options['tmp_folder'], 'r' + 'FLAIR.nii.gz'))
            # print(arg1)
            # os.system(arg1)
            # print("> INFO:", scan, "elapsed time: ", round(time.time() - reg_time), "sec")

            # ////////

            print('Path to ref data:', options['le_path'])
            for mod in options['modalities']:
                if options['rf'] is True:
                    print("registering using ", CYELLOW + "fsl" + CEND, mod, '---> MNI152_T1_1mm.nii.gz')
                    print('Flirt path:', CYELLOW + which('flirt') + CEND)
                    # print('Flirt path:', CYELLOW + os.system('which('flirt')') + CEND)
                    shutil.copy(os.path.join(options['le_path']),
                                os.path.join(options['tmp_folder'], 'MNI152_T1_1mm.nii.gz'))
                    folder_path = os.path.join(os.path.join(options['tmp_folder']))
                    os.chdir(folder_path)
                    os.system(
                        'flirt -in FLAIR.nii.gz -ref ./MNI152_T1_1mm.nii.gz -out rFLAIR.nii.gz -omat Flair_reg.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear')
                else:
                    if which('reg_aladin') is not None:
                        print("Using installed reg_aladin!")
                        print("registering using ", CYELLOW + "reg_aladin" + CEND,  mod, '---> MNI152_T1_1mm.nii.gz')
                        subprocess.check_output(['reg_aladin', '-ref',
                                             os.path.join(options['le_path']),
                                             '-flo', os.path.join(options['tmp_folder'], mod + '.nii.gz'),
                                             '-aff', os.path.join(options['tmp_folder'], mod + '_transf.txt'),
                                             '-res', os.path.join(options['tmp_folder'], 'r' + mod + '.nii.gz')])
                    else:
                        print("Using CNN reg_aladin!")
                        os_host = platform.system()
                        reg_exe = ''
                        if os_host == 'Windows':
                            reg_exe = 'reg_aladin.exe'
                        elif os_host == 'Linux' or 'Darwin':
                            reg_exe = 'reg_aladin'
                        else:
                            print("> ERROR: The OS system", os_host, "is not currently supported.")
                        reg_aladin_path = ''

                        if os_host == 'Windows':
                            reg_aladin_path = os.path.join(options['niftyreg_path'], reg_exe)
                        elif os_host == 'Linux':
                            reg_aladin_path = os.path.join(options['niftyreg_path'], reg_exe)
                        elif os_host == 'Darwin':
                            reg_aladin_path = os.path.join(options['niftyreg_path'], reg_exe)
                        else:
                            print('Please install first  NiftyReg in your mac system and try again!')
                            sys.stdout.flush()
                            time.sleep(1)
                            os.kill(os.getpid(), signal.SIGTERM)

                        print("registering using ", CYELLOW + "reg_aladin" + CEND,  mod, '---> MNI152_T1_1mm.nii.gz')
                        subprocess.check_output([reg_aladin_path, '-ref',
                                             os.path.join(options['le_path']),
                                             '-flo', os.path.join(options['tmp_folder'], mod + '.nii.gz'),
                                             '-aff', os.path.join(options['tmp_folder'], mod + '_transf.txt'),
                                             '-res', os.path.join(options['tmp_folder'], 'r' + mod + '.nii.gz')])

                # if options['use_of_fsl']:
                #     print("registering (using fsl) to", mod, " --> MNI152_T1_1mm.nii.gz space")
                #     print('Path to reg data:', options['le_path'])
                #     print('Path to input:', os.path.join(options['tmp_folder'], mod + '.nii.gz'))
                #     print('Path to output:', os.path.join(options['tmp_folder'], 'r' + mod + '.nii.gz'))
                #     subprocess.check_output([flirt,
                #                          'in', os.path.join(options['tmp_folder'], mod + '.nii.gz'),
                #                          '-ref', os.path.join(options['le_path']),
                #                          '-out', os.path.join(options['tmp_folder'], 'r' + mod + '.nii.gz'),
                #                          '-omat', os.path.join(options['tmp_folder'], 'Flair_reg.mat'),
                #                          '-bins', '256',
                #                          '-cost', 'corratio',
                #                          '-searchrx', '-90 90',
                #                          '-searchry', '-90 90',
                #                          '-searchrz', '-90 90',
                #                          '-dof', '12',
                #                          '-interp', 'trilinear'])




            print(CBLUE2 + "Registration completed!" + CEND)
            print('skull stripping!')
            if options['rf'] is True:
                print("skull stripping using:", CYELLOW + "fsl" + CEND)
            # os.system('/usr/local/fsl/bin/bet Flair_reg Flair_reg_brain  -f 0.5 -g 0') <---sec
                print('Bet path:', CYELLOW + which('bet') + CEND)
                print('Bet2 path:', CYELLOW + which('bet2') + CEND)
                print('fsl5.0-bet2 path:', CYELLOW + which('fsl5.0-bet2') + CEND)
                bet = 'fsl5.0-bet2'
                t1_im = os.path.join(options['tmp_folder'], 'rFLAIR.nii.gz')
                t1_st_im = os.path.join(options['tmp_folder'], 'FLAIR_t_x.nii.gz')
                subprocess.check_output([bet,
                                     t1_im,
                                     t1_st_im, '-f', '0.5', '-g', '0'])
            else:
                print("skull stripping using:", CYELLOW + "robex" + CEND)
                t1_im = os.path.join(options['tmp_folder'], 'rFLAIR.nii.gz')
                t1_st_im = os.path.join(options['tmp_folder'], 'FLAIR_t_x.nii.gz')
                print('Robex path:', CYELLOW + options['robex_path'] + CEND)
                out = subprocess.check_output(["bash", options['robex_path'],
                                           t1_im,
                                           t1_st_im])

            print(CBLUE2 + "skull stripping completed!" + CEND)
            print('bias correction!')
            BIAS = ['FLAIR']
            if options['bias_choice'] == 'All':
                BIAS = options['modalities']
            if options['bias_choice'] == 'FLAIR':
                BIAS = ['FLAIR']
            if options['bias_choice'] == 'T1':
                BIAS = ['T1']
            if options['bias_choice'] == 'MOD3':
                BIAS = ['MOD3']
            if options['bias_choice'] == 'MOD4':
                BIAS = ['MOD4']
            for mod in BIAS:
            # os.system('python ./N4BiasFieldCorrection-master/N4BiasFieldCorrection_FL.py') third

                input_scan = 'FLAIR_t_x.nii.gz'
                print('input_scan', input_scan)
                input_scan = os.path.join(options['tmp_folder'], 'FLAIR_t_x.nii.gz')
                print("N4 bias correction runs.")
                inputImage = sitk.ReadImage(input_scan)
            # maskImage = sitk.ReadImage("06-t1c_mask.nii.gz")
                maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
                sitk.WriteImage(maskImage, os.path.join(options['tmp_folder'], "Flair_reg_brain_bias_mask.nii.gz"))

                inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)

                corrector = sitk.N4BiasFieldCorrectionImageFilter()

                output = corrector.Execute(inputImage, maskImage)
                sitk.WriteImage(output, os.path.join(options['tmp_folder'], mod + '_n4.nii.gz'))
                print("Finished N4 Bias Field Correction for", mod, 'finished!')

            print(CBLUE2 + "bias correction completed!" + CEND)
            print(CBLUE2 + "Denoising started... reducing noise using anisotropic Diffusion" + CEND)
            denoise_time = time.time()
            tmp_scan = nib.load(os.path.join(options['tmp_folder'],
                                             'FLAIR_n4.nii.gz'))

            tmp_scan.get_data()[:] = ans_dif(tmp_scan.get_data(),
                                             niter=options['denoise_iter'])

            tmp_scan.to_filename(os.path.join(options['tmp_folder'],
                                              'FLAIR_tmp.nii.gz'))

            print("> INFO: denoising", scan, "elapsed time: ", round(time.time() - denoise_time), "sec")
            print(CBLUE2 + "Denoising completed!" + CEND)


    else:
        print(CYELLOW + "CNN standard preprocessing will be proceeded!" + CEND)

        # --------------------------------------------------
        # bias_correction(options)
        if options['bias_correction'] is True:
            denoise_time = time.time()
            bias_correction(options)
            print("> INFO: bias correction", scan, "elapsed time: ", round(time.time() - denoise_time), "sec")
        else:
            pass

        # --------------------------------------------------

        if options['register_modalities'] is True:
            print(CBLUE2 + "Registration started... moving all images to the MPRAGE+192 space" + CEND)
            reg_time = time.time()
            register_masks(options)
            print("> INFO:", scan, "elapsed time: ", round(time.time() - reg_time), "sec")
            print(CBLUE2 + "Registration completed!" + CEND)
        else:
            try:
                if options['reg_space'] == 'FlairtoT1':
                    for mod in options['modalities']:
                        if mod == 'T1':
                            continue
                        out_scan = mod + '.nii.gz' if mod == 'T1' else 'r' + mod + '.nii.gz'
                        shutil.copy2(os.path.join(options['tmp_folder'],
                                                  mod + '.nii.gz'),
                                     os.path.join(options['tmp_folder'],
                                                  out_scan))
                if options['reg_space'] == 'T1toFlair':
                    for mod in options['modalities']:
                        if mod == 'FLAIR':
                            continue
                        out_scan = mod + '.nii.gz' if mod == 'FLAIR' else 'r' + mod + '.nii.gz'
                        shutil.copy2(os.path.join(options['tmp_folder'],
                                                  mod + '.nii.gz'),
                                     os.path.join(options['tmp_folder'],
                                                  out_scan))
                if options['reg_space'] != 'FlairtoT1' and options['reg_space'] != 'T1toFlair':
                    for mod in options['modalities']:
                        out_scan = 'r' + mod + '.nii.gz'
                        shutil.copy2(os.path.join(options['tmp_folder'],
                                                  mod + '.nii.gz'),
                                     os.path.join(options['tmp_folder'],
                                                  out_scan))


            except:
                print("> ERROR: registration ", scan,
                      "I can not rename input modalities as tmp files. Quiting program.")

                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

        # --------------------------------------------------
        # noise filtering
        # --------------------------------------------------
        if options['denoise'] is True:
            print(CBLUE2 + "Denoising started... reducing noise using anisotropic Diffusion" + CEND)
            denoise_time = time.time()
            denoise_masks(options)
            print("> INFO: denoising", scan, "elapsed time: ", round(time.time() - denoise_time), "sec")
            print(CBLUE2 + "Denoising completed!" + CEND)
        else:
            try:
                for mod in options['modalities']:
                    if options['reg_space'] == 'FlairtoT1':
                        input_scan = mod + '.nii.gz' if mod == 'T1' else 'r' + mod + '.nii.gz'
                    if options['reg_space'] == 'T1toFlair':
                        input_scan = mod + '.nii.gz' if mod == 'FLAIR' else 'r' + mod + '.nii.gz'
                    if options['reg_space'] != 'FlairtoT1' and options['reg_space'] != 'T1toFlair':
                        input_scan = 'r' + mod + '.nii.gz'
                    shutil.copy(os.path.join(options['tmp_folder'],
                                             input_scan),
                                os.path.join(options['tmp_folder'],
                                             'd' + input_scan))
            except:
                print("> ERROR denoising:", scan, "I can not rename input modalities as tmp files. Quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

        # --------------------------------------------------
        # skull strip
        # --------------------------------------------------

        if options['skull_stripping'] is True:
            print(CBLUE2 + "External skull stripping started... using ROBEX or BET(Brain Extraction Tool)" + CEND)
            sk_time = time.time()
            skull_strip(options)
            print("> INFO:", scan, "elapsed time: ", round(time.time() - sk_time), "sec")
            print(CBLUE2 + "External skull stripping completed!" + CEND)

        else:
            try:
                for mod in options['modalities']:
                    if options['reg_space'] == 'FlairtoT1':
                        input_scan = 'd' + mod + '.nii.gz' if mod == 'T1' else 'dr' + mod + '.nii.gz'
                    if options['reg_space'] == 'T1toFlair':
                        input_scan = 'd' + mod + '.nii.gz' if mod == 'FLAIR' else 'dr' + mod + '.nii.gz'
                    if options['reg_space'] != 'FlairtoT1' and options['reg_space'] != 'T1toFlair':
                        input_scan = 'dr' + mod + '.nii.gz'
                    shutil.copy(os.path.join(options['tmp_folder'],
                                             input_scan),
                                os.path.join(options['tmp_folder'],
                                             mod + '_tmp.nii.gz'))
            except:
                print("> ERROR: Skull-stripping", scan,
                      "I can not rename input modalities as tmp files. Quiting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

        if options['skull_stripping'] is True and options['register_modalities'] is True:
            print("> INFO:", scan, "total preprocessing time: ", round(time.time() - preprocess_time))
