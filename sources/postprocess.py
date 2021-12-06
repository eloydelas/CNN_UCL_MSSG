import os
import signal
import subprocess
import time
import platform
import sys
import shutil
from shutil import which
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

def invert_registration(current_folder, options):

    try:

        if which('reg_transform') is not None:
            print(CRED2 + "Using installed reg_transform!" + CEND)
            print("Computing the inverse transformation using:", CYELLOW + "reg_transform " + CEND,
                  CRED + "[Step 1]" + CEND)
            print('Reading input file:', os.path.join(options['tmp_folder'],
                                                      'FLAIR_transf.txt'))
            process = subprocess.Popen(['reg_transform', '-invAff',
                                        os.path.join(options['tmp_folder'],
                                                     'FLAIR_transf.txt'),
                                        os.path.join(options['tmp_folder'],
                                                     'inv_FLAIR_transf.txt')],
                                       stdout=subprocess.PIPE,
                                       universal_newlines=True)
            print('Output file:', os.path.join(options['tmp_folder'],
                                               'inv_FLAIR_transf.txt'))

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


        else:
            print("Using CNN reg_resample!")
            # rigid registration
            reg_transform = ''
            reg_resample = ''
            os_host = platform.system()
            if os_host == 'Windows':
                reg_transform = 'reg_transform.exe'
                reg_resample = 'reg_resample.exe'
            elif os_host == 'Linux' or 'Darwin':
                reg_transform = 'reg_transform'
                reg_resample = 'reg_resample'
            else:
                print("> ERROR: The OS system", os_host, "is not currently supported.")
            reg_transform_path = ''
            reg_resample_path = ''

            if os_host == 'Windows':
                reg_transform_path = os.path.join(options['niftyreg_path'], reg_transform)
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_resample)
            elif os_host == 'Linux':
                reg_transform_path = os.path.join(options['niftyreg_path'], reg_transform)
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_resample)
            elif os_host == 'Darwin':
                reg_transform_path = reg_transform
                reg_resample_path = reg_resample
            else:
                print('Please install first  NiftyReg in your mac system and try again!')
                sys.stdout.flush()
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)
            print('running ....> ', reg_transform_path)
            print('running ....> ', reg_resample_path)

            print("Computing the inverse transformation using:", CYELLOW + "reg_transform " + CEND,
                  CRED + "[Step 1]" + CEND)
            print('Reading input file:', os.path.join(options['tmp_folder'],
                                                      'FLAIR_transf.txt'))
            process = subprocess.Popen([reg_transform_path, '-invAff',
                                        os.path.join(options['tmp_folder'],
                                                     'FLAIR_transf.txt'),
                                        os.path.join(options['tmp_folder'],
                                                     'inv_FLAIR_transf.txt')],
                                       stdout=subprocess.PIPE,
                                       universal_newlines=True)
            print('Output file:', os.path.join(options['tmp_folder'],
                                               'inv_FLAIR_transf.txt'))

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
        print("> ERROR: computing the inverse transformation matrix.\
        Quitting program.")
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

    print("> POST: registering output segmentation masks back to FLAIR")

    current_experiment = os.path.join(current_folder, options['experiment'])
    list_scans = os.listdir(current_experiment)

    try:
        if which('reg_resample') is not None:
            print(CRED2 + "Using installed reg_resample!" + CEND)
            print("Computing the inverse transformation using:", CYELLOW + "reg_resample " + CEND, CRED + "[Step 2]" + CEND)
            file = options['experiment'] + '_CNN_final_segmentation.nii.gz'
            print('Processing the file:', file)
            if options['use_of_fsl'] is True:
                print('using FLAIR.nii.gz as reference!')
                process = subprocess.Popen(['reg_resample',
                                            '-ref', os.path.join(options['tmp_folder'],
                                                                 'FLAIR.nii.gz'),
                                            '-flo', os.path.join(current_experiment,
                                                                 file),
                                            '-trans', os.path.join(options['tmp_folder'],
                                                                   'inv_FLAIR_transf.txt'),
                                            '-res', os.path.join(current_experiment,
                                                                 'Mask_BacktoOriginalSpace.nii.gz'),
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
            else:
                print('using FLAIR.nii.gz as reference!')
                process = subprocess.Popen(['reg_resample',
                                            '-ref', os.path.join(options['tmp_folder'],
                                                                 'FLAIR.nii.gz'),
                                            '-flo', os.path.join(current_experiment,
                                                                 file),
                                            '-trans', os.path.join(options['tmp_folder'],
                                                                   'inv_FLAIR_transf.txt'),
                                            '-res', os.path.join(current_experiment,
                                                                 'Mask_BacktoOriginalSpace.nii.gz'),
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





            print('\x1b[6;30;42m' + '......................[Applying final transformations done!]......................' + '\x1b[0m')
            print('\x1b[6;30;42m' + 'The predicted segmentation output can be found in the Model* folder under the name' + '\x1b[0m')
            print('\x1b[6;30;42m' + '..........................Mask_BacktoOriginalSpace.nii.gz.........................' + '\x1b[0m')


        else:
            print("Using CNN reg_resample!")
            # rigid registration
            reg_transform = ''
            reg_resample = ''
            os_host = platform.system()
            if os_host == 'Windows':
                reg_transform = 'reg_transform.exe'
                reg_resample = 'reg_resample.exe'
            elif os_host == 'Linux' or 'Darwin':
                reg_transform = 'reg_transform'
                reg_resample = 'reg_resample'
            else:
                print("> ERROR: The OS system", os_host, "is not currently supported.")
            reg_transform_path = ''
            reg_resample_path = ''

            if os_host == 'Windows':
                reg_transform_path = os.path.join(options['niftyreg_path'], reg_transform)
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_resample)
            elif os_host == 'Linux':
                reg_transform_path = os.path.join(options['niftyreg_path'], reg_transform)
                reg_resample_path = os.path.join(options['niftyreg_path'], reg_resample)
            elif os_host == 'Darwin':
                reg_transform_path = reg_transform
                reg_resample_path = reg_resample
            else:
                print('Please install first  NiftyReg in your mac system and try again!')
                sys.stdout.flush()
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)
            print("Computing the inverse transformation using:", CYELLOW + "reg_resample " + CEND, CRED + "[Step 2]" + CEND)
            file = options['experiment'] + '_CNN_final_segmentation.nii.gz'
            print('Processing the file:', file)
            if options['use_of_fsl'] is True:
                print('using FLAIR.nii.gz as reference!')
                process = subprocess.Popen([reg_resample_path,
                                            '-ref', os.path.join(options['tmp_folder'],
                                                                 'FLAIR.nii.gz'),
                                            '-flo', os.path.join(current_experiment,
                                                                 file),
                                            '-trans', os.path.join(options['tmp_folder'],
                                                                   'inv_FLAIR_transf.txt'),
                                            '-res', os.path.join(current_experiment,
                                                                 'Mask_BacktoOriginalSpace.nii.gz'),
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
            else:
                print('using FLAIR.nii.gz as reference!')
                process = subprocess.Popen([reg_resample_path,
                                            '-ref', os.path.join(options['tmp_folder'],
                                                                 'FLAIR.nii.gz'),
                                            '-flo', os.path.join(current_experiment,
                                                                 file),
                                            '-trans', os.path.join(options['tmp_folder'],
                                                                   'inv_FLAIR_transf.txt'),
                                            '-res', os.path.join(current_experiment,
                                                                 'Mask_BacktoOriginalSpace.nii.gz'),
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





            print('\x1b[6;30;42m' + '......................[Applying final transformations done!]......................' + '\x1b[0m')
            print('\x1b[6;30;42m' + 'The predicted segmentation output can be found in the Model* folder under the name' + '\x1b[0m')
            print('\x1b[6;30;42m' + '..........................Mask_BacktoOriginalSpace.nii.gz.........................' + '\x1b[0m')




    except:
            print("> ERROR: resampling CNN_final_segmentation", "Quitting program.")
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)
