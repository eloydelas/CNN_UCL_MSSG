ARG UBUNTU_VERSION=18.04
# FROM neurodebian:trusty
FROM neurodebian:xenial
# ARG ARCH=
# ARG CUDA=10.0
# FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARG ARCH
# ARG CUDA
# ARG CUDNN=7.6.2.24-1

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         cuda-command-line-tools-${CUDA/./-} \
#         cuda-cublas-${CUDA/./-} \
#         cuda-cufft-${CUDA/./-} \
#         cuda-curand-${CUDA/./-} \
#         cuda-cusolver-${CUDA/./-} \
#         cuda-cusparse-${CUDA/./-} \
#         curl \
#         libcudnn7=${CUDNN}+cuda${CUDA} \
#         libfreetype6-dev \
#         libhdf5-serial-dev \
#         libzmq3-dev \
#         pkg-config \
#         software-properties-common \
#         unzip
# RUN [ ${ARCH} = ppc64le ] || (apt-get update && \
#         apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda${CUDA} \
#         && apt-get clean \
#         && rm -rf /var/lib/apt/lists/*)
# ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
#     && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
#     && echo "/usr/local/cuda/extras/CUPTI/lib64" > /etc/ld.so.conf.d/cupti.conf \
#     && ldconfig





ARG USE_PYTHON_3_NOT_2
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}
ENV LANG C.UTF-8
RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

# RUN ${PIP} --no-cache-dir install --upgrade \
#     pip \
#     setuptools

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

# ARG TF_PACKAGE=tensorflow-gpu
# ARG TF_PACKAGE_VERSION=
# RUN ${PIP} install ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}


# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"




MAINTAINER thisgithub

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends && apt-get install -y sudo \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
#     bc \
#     build-essential \
#     curl \
#     dc \
#     git \
#     libegl1-mesa-dev \
#     libopenblas-dev \
#     nano \
#     python2.7 \
#     python3 \
#     tar \
#     tcsh \
#     tzdata \
#     unzip \
#     wget

# PPA for newer version of nodejs, which is required for bids-validator
# RUN curl -sL https://deb.nodesource.com/setup_12.x -o nodesource_setup.sh && \
#     bash nodesource_setup.sh && \
#     rm -f nodesource_setup.sh && \
#     apt-get install -y nodejs

# # NeuroDebian setup
# COPY neurodebian.gpg /neurodebian.gpg
# RUN wget -qO- http://neuro.debian.net/lists/bionic.au.full | \
#     tee /etc/apt/sources.list.d/neurodebian.sources.list && \
#     apt-key add /neurodebian.gpg && \
#     apt-get update
#
# # Additional dependencies for MRtrix3 compilation
# RUN apt-get update && apt-get install -y \
#     libeigen3-dev \
#     libfftw3-dev \
#     libpng-dev \
#     libtiff5-dev \
#     zlib1g-dev
#
# # Neuroimaging software / data dependencies
# RUN wget -qO- https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.1.1/freesurfer-linux-centos8_x86_64-7.1.1.tar.gz | \
#     tar zx -C /opt \
#     --exclude='freesurfer/trctrain' \
#     --exclude='freesurfer/subjects/fsaverage_sym' \
#     --exclude='freesurfer/subjects/fsaverage3' \
#     --exclude='freesurfer/subjects/fsaverage4' \
#     --exclude='freesurfer/subjects/fsaverage6' \
#     --exclude='freesurfer/subjects/cvs_avg35' \
#     --exclude='freesurfer/subjects/cvs_avg35_inMNI152' \
#     --exclude='freesurfer/subjects/bert' \
#     --exclude='freesurfer/subjects/V1_average' \
#     --exclude='freesurfer/average/mult-comp-cor' \
#     --exclude='freesurfer/lib/cuda' \
#     --exclude='freesurfer/lib/qt'
# RUN echo "cHJpbnRmICJyb2JlcnQuc21pdGhAZmxvcmV5LmVkdS5hdVxuMjg1NjdcbiAqQ3FLLjFwTXY4ZE5rXG4gRlNvbGRZRXRDUFZqNlxuIiA+IC9vcHQvZnJlZXN1cmZlci9saWNlbnNlLnR4dAo=" | base64 -d | sh
# RUN apt-get install -y ants=2.2.0-1ubuntu1
# FSL installer appears to now be ready for use with version 6
# eddy is also now included in FSL6
# RUN wget -q http://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py && \
#     chmod 775 fslinstaller.py && \
#     python2 /fslinstaller.py -d /opt/fsl -V 6.0.4 -q && \
#     rm -f /fslinstaller.py

# RUN echo "source activate tfg" > ~/.bashrc

#
#
# RUN which immv || ( echo "FSLPython not properly configured; re-running" && rm -rf /opt/fsl/fslpython && /opt/fsl/etc/fslconf/fslpython_install.sh -f /opt/fsl || ( cat /tmp/fslpython*/fslpython_miniconda_installer.log && exit 1 ) )
# RUN wget -qO- "https://www.nitrc.org/frs/download.php/5994/ROBEXv12.linux64.tar.gz//?i_agree=1&download_now=1" | \
#     tar zx -C /opt
# RUN npm install -gq bids-validator@1.5.3
#
# # apt cleanup to recover as much space as possible
# RUN apt-get remove -y libegl1-mesa-dev && \
#     apt-get autoremove -y && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
#
# # Download additional data for neuroimaging software, e.g. templates / atlases
# RUN wget -qO- http://www.gin.cnrs.fr/AAL_files/aal_for_SPM12.tar.gz | \
#     tar zx -C /opt && \
#     wget -qO- http://www.gin.cnrs.fr/AAL2_files/aal2_for_SPM12.tar.gz | \
#     tar zx -C /opt
# #RUN wget -q http://www.nitrc.org/frs/download.php/4499/sri24_anatomy_nifti.zip -O sri24_anatomy_nifti.zip && \
# #    unzip -qq -o sri24_anatomy_nifti.zip -d /opt/ && \
# #    rm -f sri24_anatomy_nifti.zip
# #RUN wget -q http://www.nitrc.org/frs/download.php/4502/sri24_anatomy_unstripped_nifti.zip -O sri24_anatomy_unstripped_nifti.zip && \
# #    unzip -qq -o sri24_anatomy_unstripped_nifti.zip -d /opt/ && \
# #    rm -f sri24_anatomy_unstripped_nifti.zip
# #RUN wget -q http://www.nitrc.org/frs/download.php/4508/sri24_labels_nifti.zip -O sri24_labels_nifti.zip && \
# #    unzip -qq -o sri24_labels_nifti.zip -d /opt/ && \
# #    rm -f sri24_labels_nifti.zip
# RUN wget -q https://github.com/AlistairPerry/CCA/raw/master/parcellations/512inMNI.nii -O /opt/512inMNI.nii
# #RUN wget -q https://ndownloader.figshare.com/files/3133832 -O oasis.zip && \
# #    unzip -qq oasis.zip -d /opt/ && \
# #    rm -f oasis.zip
# RUN wget -qO- http://www.nitrc.org/frs/download.php/5906/ADHD200_parcellations.tar.gz | \
#     tar zx -C /opt
# RUN wget -q "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/5528816/lh.HCPMMP1.annot" \
#     -O /opt/freesurfer/subjects/fsaverage/label/lh.HCPMMP1.annot && \
#     wget -q "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/5528819/rh.HCPMMP1.annot" \
#     -O /opt/freesurfer/subjects/fsaverage/label/rh.HCPMMP1.annot
# RUN mkdir /opt/brainnetome && \
#     ( wget -q "http://ddl.escience.cn/f/IiyU?func=download&rid=8135438" -O /opt/freesurfer/average/rh.BN_Atlas.gcs || \
#     wget -q "https://osf.io/e6zkg/download" -O /opt/freesurfer/average/rh.BN_Atlas.gcs ) && \
#     ( wget -q "http://ddl.escience.cn/f/IiyP?func=download&rid=8135433" -O /opt/freesurfer/average/lh.BN_Atlas.gcs || \
#     wget -q "https://osf.io/af9ut/download" -O /opt/freesurfer/average/lh.BN_Atlas.gcs ) && \
#     ( wget -q "http://ddl.escience.cn/f/PC7Q?func=download&rid=9882718" -O /opt/freesurfer/average/BN_Atlas_subcortex.gca || \
#     wget -q "https://osf.io/k2cd8/download" -O /opt/freesurfer/average/BN_Atlas_subcortex.gca ) && \
#     ( wget -q "http://ddl.escience.cn/f/PC7O?func=download&rid=9882716" -O /opt/brainnetome/BN_Atlas_246_LUT.txt || \
#     wget -q "https://osf.io/eb7pm/download" -O /opt/brainnetome/BN_Atlas_246_LUT.txt ) && \
#     ( wget -q "http://ddl.escience.cn/f/Bvhg?func=download&rid=6516020" -O /opt/brainnetome/BNA_MPM_thr25_1.25mm.nii.gz || \
#     wget -q "https://osf.io/dbqep/download" -O /opt/brainnetome/BNA_MPM_thr25_1.25mm.nii.gz ) && \
#     cp /opt/brainnetome/BN_Atlas_246_LUT.txt /opt/freesurfer/
# RUN wget -qO- "https://github.com/ThomasYeoLab/CBIG/archive/v0.11.1-Wu2017_RegistrationFusion.tar.gz" | \
#     tar zx -C /opt && \
#     cp /opt/CBIG-0.11.1-Wu2017_RegistrationFusion/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/fsaverage5/label/*h.Yeo2011_*Networks_N1000.split_components.annot /opt/freesurfer/subjects/fsaverage5/label/ && \
#     cp /opt/CBIG-0.11.1-Wu2017_RegistrationFusion/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/project_to_individual/Yeo2011_*networks_Split_Components_LUT.txt /opt/freesurfer/ && \
#     mkdir /opt/Yeo2011 && \
#     cp /opt/CBIG-0.11.1-Wu2017_RegistrationFusion/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/MNI152/Yeo2011_*Networks_N1000.split_components.FSL_MNI152_*mm.nii.gz /opt/Yeo2011/ && \
#     cp /opt/CBIG-0.11.1-Wu2017_RegistrationFusion/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/MNI152/*Networks_ColorLUT_freeview.txt /opt/Yeo2011/ && \
#     rm -rf /opt/CBIG-0.11.1-Wu2017_RegistrationFusion
#
# # Setup envvars
# ENV ANTSPATH=/usr/lib/ants \
#     FREESURFER_HOME=/opt/freesurfer \
#     FMRI_ANALYSIS_DIR=/opt/freesurfer/fsfast \
#     FSF_OUTPUT_FORMAT=nii.gz \
#     FSFAST_HOME=/opt/freesurfer/fsfast \
#     LOCAL_DIR=/opt/freesurfer/local \
#     MINC_BIN_DIR=/opt/freesurfer/mni/bin \
#     MINC_LIB_DIR=/opt/freesurfer/mni/lib \
#     MNI_DATAPATH=/opt/freesurfer/mni/data \
#     MNI_DIR=/opt/freesurfer/mni \
#     MNI_PERL5LIB=/opt/freesurfer/mni/lib/perl5/5.8.5 \
#     OS=Linux \
#     PERL5LIB=/opt/freesurfer/mni/lib/perl5/5.8.5 \
#     SUBJECTS_DIR=/opt/freesurfer/subjects \
#     FSLDIR=/opt/fsl \
#     FSLOUTPUTTYPE=NIFTI \
#     FSLMULTIFILEQUIT=TRUE \
#     FSLTCLSH=/opt/fsl/bin/fsltclsh \
#     FSLWISH=/opt/fsl/bin/fslwish \
#     LD_LIBRARY_PATH=/opt/fsl/lib:$LD_LIBRARY_PATH \
#     PATH=/opt/mrtrix3/bin:/usr/lib/ants:/opt/freesurfer/bin:/opt/freesurfer/mni/bin:/opt/fsl/bin:/opt/ROBEX:$PATH \
#     PYTHONPATH=/opt/mrtrix3/lib:$PYTHONPATH
#


# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

# RUN wget -q http://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py && \
#     chmod 775 fslinstaller.py && \
#     python2 /fslinstaller.py -d /opt/fsl -V 6.0.4 -q && \
#     rm -f /fslinstaller.py
#
# RUN rm -rf /opt/fsl/fslpython && \
#    /opt/fsl/etc/fslconf/fslpython_install.sh -f /opt/fsl

# Install Python packages and keras
# ENV NB_USER keras
# ENV NB_UID 1000
#
# RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
#     chown $NB_USER $CONDA_DIR -R && \
#     mkdir -p /src && \
#     chown $NB_USER /src

# USER $NB_USER
USER root
RUN  pip install --upgrade pip

COPY sourcesx /etc/apt/sources.list.d/neurodebian.sources.list
RUN apt-get update \
    && apt-get install -y \
      fsl-5.0-core \
      jq

# Configure environment (Must also be done it the RUN script)
ENV FSLDIR=/usr/lib/fsl/5.0
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV PATH=$PATH:$FSLDIR
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FSLDIR
# RUN echo ". /etc/fsl/5.0/fsl.sh" >> /root/.bashrc
RUN echo ". /etc/fsl/5.0/fsl.sh" >> ~/.bashrc
RUN echo ". /etc/fsl/5.0/fsl.sh" >> ~/.bash_profile

# Make directory for flywheel spec (v0)
ENV FLYWHEEL /flywheel/v0
RUN mkdir -p ${FLYWHEEL}
COPY run ${FLYWHEEL}/run
COPY manifest.json ${FLYWHEEL}/manifest.json

# Configure entrypoint
ENTRYPOINT ["/flywheel/v0/run"]

# COPY sourcesx /etc/apt/sources.list.d/neurodebian.sources.list
# RUN apt-get update \
#     && apt-get install -y fsl-5.0-core
#
# # Configure environment
# ENV FSLDIR=/usr/lib/fsl/5.0
# ENV FSLOUTPUTTYPE=NIFTI_GZ
# ENV PATH=$PATH:$FSLDIR
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FSLDIR
#
# # Run configuration script for normal usage
# RUN echo ". /etc/fsl/5.0/fsl.sh" >> ~/.bashrc
# RUN echo ". /etc/fsl/5.0/fsl.sh" >> ~/.bash_profile















# RUN apt-get update \
#     && apt-get install -y fsl-5.0-core
#
# # Configure environmentfsl/5.0/bin/
# ENV FSLDIR=/usr/lib/fsl/5.0
# ENV FSLOUTPUTTYPE=NIFTI_GZ
# ENV PATH=$PATH:$FSLDIR
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FSLDIR

# FSL Setup
# ENV FSLDIR=~/libs/linux/fsl
# ENV PATH=${FSLDIR}/bin:${PATH}
# ENV PATH=$PATH:$FSLDIR
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FSLDIR
# RUN echo ". /home/executiveuser/libs/linux/fsl/etc/fslconf/fsl.sh" > ~/.bashrc


# Run configuration script for normal usage
# RUN echo ". /etc/fsl/5.0/fsl.sh" >> /root/.bashrc
# RUN echo ". /etc/fsl/5.0/fsl.sh" > ~/.bashrc
# copy fslinstallerx.py .
# RUN sudo apt-get update && apt-get install -y --no-install-recommends
# RUN python2.7 fslinstallerx.py -d /usr/local/fsl



        
# Install git, wget, python-dev, pip, BLAS + LAPACK and other dependencies
# RUN apt-get update && apt-get install -y \
#  gfortran \
#  liblapack-dev \
#  libopenblas-dev \
#  python-dev \
#  python-tk\
#  git \
#  curl \
  # emacs24
  




ARG python_version=3.7
ENV PATH /opt/conda/bin:$PATH
ENV PATH /opt/conda/envs/tfg/bin:$PATH
RUN conda config --add channels intel
# COPY tensorflow-gpu.yml .
# RUN conda env create -f tensorflow-gpu.yml


COPY environment.yml .
RUN conda env create -f environment.yml

#     conda create -n idp intelpython3_full python=3
# RUN conda create -n idp python=3.7
RUN echo "source activate tfg" > ~/.bashrc
# RUN pip install numpy==1.16.3
ENV PATH /opt/conda/envs/tfg/bin:$PATH

# ADD libs/linux/fsl/etc/fslconf/* .
# COPY libs/linux/fsl/etc/fslconf/fsl.sh .
# # RUN echo ". /fsl.sh" > ~/.bashrc
# # RUN echo ". /fsl.sh" > ~/.bash_profile
# #     conda create -n idp intelpython3_full python=3
# RUN echo "source fsl.sh" >> ~/.bashrc
# RUN echo "source fsl.sh" >> ~/.bash_profile
# Activate the environment, and make sure it's activated:
# RUN conda activate tensorflow-gpu
# RUN echo "The environment tensorflow-gpu is installed!"

# RUN echo "source activate idp" > ~/.bashrc
# RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Install miniconda to /miniconda
# RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
# RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
# RUN rm /Miniconda-latest-Linux-x86_64.sh
# ENV PATH=/opt/conda/bin:${PATH}

# ENV PATH=/miniconda/envs/idp/bin:$PATH
# RUN conda remove -n tensorflow
# ARG python_version=3.6



# RUN pip install \
#       medpy \
#       nibabel \
#       numpy \
#       pillow \
#       simpleitk \
#       pydicom \
#       pyyaml \
#       scipy \
#       Keras==2.3.1 \
#       tensorflow==2.0.0 \
#       tensorboard \
#       tensorflow-estimator \
#       tensorflow-gpu \
#       sklearn_pandas \
#       h5py \
#       MedPy \
#       click \
#       signals && \
#     conda install \
#       bzip2 \
#       certifi \
#       intelpython \
#       libgcc-ng \
#       openssl \
#       pip \
#       python \
#       setuptools \
#       sqlite \
#       tcl \
#       tk \
#       wheel \
#       xz \
#       zlib \
#       bcolz \
#       h5py \
#       matplotlib \
#       mkl \
#       nose \
#       notebook \
#       pandas \
#       pydot \
#       pygpu \
#       pyyaml \
#       scikit-learn \
#       six \
#       theano \
#       pygpu \
#       && \
# #     git clone git://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
# #     pip install git+git://github.com/keras-team/keras.git && \
#     conda clean -yt






# install CNN related packages

# RUN conda install numpy scipy mkl
# RUN conda install theano pygpu
# RUN pip install pip --upgrade
# RUN pip install -r /requirements.txt
# RUN pip uninstall protobuf
# RUN conda install tensorflow-gpu
# create a docker user
# RUN useradd -ms /bin/bash docker
# ENV HOME /home/docker
RUN useradd -ms /bin/bash executiveuser
ENV HOME /home/executiveuser
USER executiveuser
WORKDIR /home/executiveuser
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# USER newuser
# WORKDIR /home/executiveuser
# RUN mkdir CNN_SM
# RUN cd CNN_SM
# WORKDIR /CNN_SM
# WORKDIR /CNN_SM/executiveuser
# ADD codes /root/test
# USER docker
# WORKDIR $HOME/src

# copy necessary files to container
# RUN mkdir $HOME/CNN_SM
# ENV PATH=/$HOME/CNN_SM:${PATH}
ADD __init__.py $HOME
RUN mkdir libs
ADD libs $HOME/libs
RUN mkdir ssh
ADD ssh $HOME/ssh
RUN mkdir sources
ADD sources $HOME/sources
RUN mkdir nets
ADD nets $HOME/nets

RUN mkdir config
ADD config $HOME/config

ADD CNN.jpeg $HOME
ADD predictor.py $HOME


RUN mkdir Singularity_le
ADD Singularity_le $HOME/Singularity_le


ADD logonic.png $HOME
ADD training_script.py $HOME
ADD CNN_GUI.py $HOME
ADD inference_script.py $HOME
ADD preprocess_inference_script.py $HOME
ADD cnn_main.py $HOME
RUN mkdir tensorboardlogs
ADD tensorboardlogs $HOME/tensorboardlogs
ADD requirements.txt $HOME

# CMD ["python", "-u", "/home/executiveuser/predictor.py"]
# RUN ["/bin/bash", "-c", "source /home/executiveuser/libs/linux/fsl/etc/fslconf/fsl.sh"]

CMD ["conda", "run", "-n", "tfg", "python", "-u", "/home/executiveuser/predictor.py"]
# add permissions (odd)
# RUN chown docker -R nets
# RUN chown docker -R config


