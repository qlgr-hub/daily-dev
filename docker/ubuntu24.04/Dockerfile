# ref:https://github.com/NVIDIA/TensorRT/blob/release/10.0/docker/ubuntu-22.04.Dockerfile
FROM nvcr.io/nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

# timezone settings
ENV TIME_ZONE=Asia/Shanghai 
RUN ln -snf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime && echo $TIME_ZONE > /etc/timezone


# Install necessary tools
RUN apt update && apt install -y --no-install-recommends sudo vim net-tools openssh-server wget curl aria2 git zip unzip rsync golang     \
                                build-essential ninja-build software-properties-common lsb-release pkg-config python3 python3-dev clinfo  \
                                python3-pip python3-wheel python3-setuptools openssl doxygen pocl-opencl-icd nodejs default-jre graphviz  \
                                libssl-dev libxml2-dev libjsoncpp-dev libomp-dev libpng-dev libtiff-dev zlib1g-dev libedit-dev libstb-dev \
                                libsfml-dev libglew-dev libglm-dev libtclap-dev


# Install ffmpeg development related packages
RUN apt update && apt install -y --no-install-recommends \
                   autoconf automake git-core libass-dev libfreetype6-dev libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev \
                   libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev texinfo libasound2-dev libgl1-mesa-dev libglew-dev libglm-dev    \
                   nasm libx264-dev libx265-dev libvpx-dev libfdk-aac-dev libmp3lame-dev libopus-dev iputils-ping


# For RISC-V dev 
RUN apt install -y device-tree-compiler libboost-regex-dev libboost-system-dev
                                

# Delete default non-root user
RUN deluser --remove-home ubuntu

# Add default non-root sudoer user and start ssh service
RUN groupadd -r -g 1000 george && useradd -rm -d /home/george -s /bin/bash -g george -G sudo -u 1000 george
RUN echo 'george:123456' | chpasswd && echo "george ALL=(ALL) ALL" >> /etc/sudoers
RUN echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config && service ssh start


# my own proxy settings
# ENV http_proxy=http://192.168.3.126:7890
# ENV https_proxy=http://192.168.3.126:7890

USER root


# Install llvm
WORKDIR /tmp
RUN aria2c https://apt.llvm.org/llvm.sh  && chmod +x llvm.sh && ./llvm.sh 20
# ENV CC=clang-20
# ENV CXX=clang++-20


# Install TensorRT
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz \
    && tar -xf TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz \
    && cp -a TensorRT-10.7.0.23/lib/*.so* /usr/lib/x86_64-linux-gnu 


# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set


# Install cmake
RUN aria2c https://github.com/Kitware/CMake/releases/download/v3.31.5/cmake-3.31.5-linux-x86_64.sh && \
    bash cmake-3.31.5-linux-x86_64.sh --skip-license --prefix=/usr/local



RUN apt update && apt install -y python3.12-venv

USER george

# Create venv and active it
RUN mkdir ~/.venv && python3 -m venv ~/.venv

# Install PyPI packages
RUN . ~/.venv/bin/activate && pip3 install --upgrade pip
RUN . ~/.venv/bin/activate && pip3 install TensorRT-10.7.0.23/python/tensorrt-10.7.0-cp312-none-linux_x86_64.whl
RUN . ~/.venv/bin/activate && pip3 install setuptools>=41.0.0
ADD requirements.txt /tmp/requirements.txt
RUN . ~/.venv/bin/activate && pip3 install -r  /tmp/requirements.txt


USER root

# Set environment
ENV PATH="${PATH}:/usr/local/bin/ngc-cli"


# Clean up
RUN rm -rf /tmp/* && apt autopurge -y


EXPOSE 22
EXPOSE 8888
WORKDIR /home/george
ADD startup.sh startup.sh
ENTRYPOINT ["bash", "startup.sh"]
