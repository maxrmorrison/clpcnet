FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# Apt dependencies
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    gcc-multilib \
    libsndfile1 \
    make \
    sox \
    wget

# Conda setup (from continuumio/miniconda3 image)
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    /opt/conda/bin/conda create -n clpcnet python=3.7 -y && \
    echo "conda activate clpcnet" >> ~/.bashrc

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "clpcnet", "/bin/bash", "-c"]

# Conda environment setup
RUN conda install -c anaconda cudatoolkit=10.0 cudnn=7.6 -y

# Allow users to specify a directory for HTK
ARG HTK=htk

# Setup htk
COPY $HTK /htk
WORKDIR /htk
RUN ./configure --disable-hslab && make all && make install

# Copy python setup files
COPY requirements.txt /clpcnet/requirements.txt

# Install python dependencies
WORKDIR /clpcnet
RUN pip install -r requirements.txt

# Copy C preprocessing code
COPY Makefile /clpcnet/Makefile
COPY src /clpcnet/src

# Build C preprocessing code
RUN make

# Copy module
COPY README.md /clpcnet/README.md
COPY setup.py /clpcnet/setup.py
COPY clpcnet /clpcnet/clpcnet

# Install module
RUN pip install -e .

# Start bash shell when run
CMD ["bash"]
