# ref:https://github.com/NVIDIA/TensorRT/blob/release/10.0/docker/ubuntu-22.04.Dockerfile
ARG CUDA_VERSION=12.2.2
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive
# timezone settings
ENV TIME_ZONE=Asia/Shanghai 
RUN ln -snf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime && echo $TIME_ZONE > /etc/timezone

# Install cudnn
ENV NV_CUDNN_VERSION=8.9.6.50
ENV NV_CUDNN_PACKAGE_NAME="libcudnn8"
ENV CUDA_VERSION_MAJOR_MINOR=12.2
ENV NV_CUDNN_PACKAGE="libcudnn8=$NV_CUDNN_VERSION-1+cuda${CUDA_VERSION_MAJOR_MINOR}"
ENV NV_CUDNN_PACKAGE_DEV="libcudnn8-dev=$NV_CUDNN_VERSION-1+cuda${CUDA_VERSION_MAJOR_MINOR}"
ENV TRT_VERSION=10.0.1.6
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*


# Update CUDA signing key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub


# Install necessary tools
RUN apt update && apt install -y --no-install-recommends sudo vim net-tools openssh-server wget curl aria2 git zip unzip rsync golang \
                                build-essential ninja-build software-properties-common lsb-release pkg-config python3 python3-dev clinfo \
                                python3-pip python3-wheel python3-setuptools openssl doxygen pocl-opencl-icd nodejs default-jre graphviz \
                                libssl-dev libxml2-dev libjsoncpp-dev libomp-dev libpng-dev libtiff-dev zlib1g-dev libedit-dev libstb-dev \
                                libsfml-dev libglew-dev libglm-dev libtclap-dev


# Install ffmpeg development related packages
RUN apt update && apt install -y --no-install-recommends \
                   autoconf automake git-core libass-dev libfreetype6-dev libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev \
                   libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev texinfo libasound2-dev libgl1-mesa-dev libglew-dev libglm-dev \
                   nasm libx264-dev libx265-dev libvpx-dev libfdk-aac-dev libmp3lame-dev libopus-dev iputils-ping
                                

# Add default non-root sudoer user and start ssh service
RUN groupadd -r -g 1000 george && useradd -rm -d /home/george -s /bin/bash -g george -G sudo -u 1000 george
RUN echo 'george:123456' | chpasswd && echo "george ALL=(ALL) ALL" >> /etc/sudoers
RUN echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config && service ssh start


# my own proxy settings
ENV http_proxy=http://192.168.3.126:7890
ENV https_proxy=http://192.168.3.126:7890

USER root


# Install llvm
WORKDIR /tmp
RUN aria2c https://apt.llvm.org/llvm.sh  && chmod +x llvm.sh && ./llvm.sh 18
# ENV CC=clang-18
# ENV CXX=clang++-18


# Install TensorRT
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz \
    && tar -xf TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz \
    && cp -a TensorRT-10.0.1.6/lib/*.so* /usr/lib/x86_64-linux-gnu \
    && pip install TensorRT-10.0.1.6/python/tensorrt-10.0.1-cp310-none-linux_x86_64.whl


# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set


# Install cmake
RUN aria2c https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0-linux-x86_64.sh && \
    bash cmake-3.30.0-linux-x86_64.sh --skip-license --prefix=/usr/local


# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r  /tmp/requirements.txt


# Set environment and working directory
ENV TRT_LIBPATH=/usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH=/workspace/daily/opensource/TensorRT
ENV PATH="${PATH}:/usr/local/bin/ngc-cli"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/out:${TRT_LIBPATH}"


# Clean up
RUN rm -rf /tmp/* && apt autopurge -y


EXPOSE 22
EXPOSE 8888
WORKDIR /home/george
ADD startup.sh startup.sh
ENTRYPOINT ["bash", "startup.sh"]
