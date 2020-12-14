FROM nvidia/cuda:10.0-cudnn7-devel-centos7
MAINTAINER Gabriel Fabien-Ouellet <gabriel.fabien-ouellet@polymtl.ca>

RUN yum -y install epel-release
RUN yum-config-manager --enable epel 
RUN yum -y install hdf5-devel \
    && yum -y install make \
    && yum -y install git 
ENV CUDA_PATH /usr/local/cuda

RUN git clone https://github.com/gfabieno/SeisCL.git
RUN cd SeisCL/src \
    && make all api=cuda nompi=1 H5CC=gcc

ENV PATH="/SeisCL/src:${PATH}"
RUN yum install -y python36 python36-devel
RUN pip3 install tensorflow-gpu==1.14.0 \
    && pip3 install scipy==1.2.0\
    && pip3 install hdf5storage==0.1.15\
    && pip3 install matplotlib==3.0.2\
    && cd /SeisCL && pip3 install .
