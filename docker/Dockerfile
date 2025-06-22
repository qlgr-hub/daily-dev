FROM nvcr.io/nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

# Timezone settings
ENV TIME_ZONE=Asia/Shanghai 
RUN ln -snf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime && echo $TIME_ZONE > /etc/timezone


# My own proxy settings
ENV http_proxy=http://192.168.3.126:7890
ENV https_proxy=http://192.168.3.126:7890


# Install necessary tools
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update && apt install -y --no-install-recommends  \
    sudo rsync vim wget curl aria2 ca-certificates gnupg net-tools iputils-ping openssl openssh-server             \
    git git-core  lsb-release pkg-config zip unzip bzip2 pbzip2 graphviz golang nodejs default-jre pv              \
    build-essential ninja-build doxygen nasm ccache device-tree-compiler libboost-system-dev libboost-regex-dev    \
    python3 python3-dev python3-pip python3-wheel python3-setuptools python3-venv clinfo pocl-opencl-icd           \
    libssl-dev libcurl4-openssl-dev libjsoncpp-dev libxml2-dev zlib1g-dev libomp-dev libopenmpi-dev libxrender-dev \
    libedit-dev libstb-dev libsfml-dev libglew-dev libglm-dev libtclap-dev libffi-dev libglib2.0-0 liblapack-dev   \
    libsm6 libxext6 libopenblas-dev libtiff-dev libpng-dev libjpeg-dev lintian fakeroot dh-make devscripts


# Install ffmpeg development related packages
RUN apt update && apt install -y --no-install-recommends \
    autoconf automake  libass-dev libfreetype6-dev libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev      \
    libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev texinfo libasound2-dev libgl1-mesa-dev libglew-dev libglm-dev \
    libx264-dev libx265-dev libvpx-dev libfdk-aac-dev libmp3lame-dev libopus-dev
                                

# Delete default non-root user
RUN deluser --remove-home ubuntu

# Add default non-root sudoer user and start ssh service
RUN groupadd -r -g 1000 george && useradd -rm -d /home/george -s /bin/bash -g george -G sudo -u 1000 george
RUN echo 'george:123456' | chpasswd && echo "george ALL=(ALL) ALL" >> /etc/sudoers
RUN echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config && service ssh start


USER root
RUN cd /usr/local/bin && ln -s /usr/bin/python3 python && ln -s /usr/bin/pip3 pip
WORKDIR /tmp


# Install TensorRT
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz \
    && tar -xf TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz \
    && cp -a TensorRT-10.7.0.23/lib/*.so* /usr/lib/x86_64-linux-gnu 


# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set


# Install cmake
RUN aria2c https://github.com/Kitware/CMake/releases/download/v4.0.3/cmake-4.0.3-linux-x86_64.sh && \
    bash cmake-4.0.3-linux-x86_64.sh --skip-license --prefix=/usr/local

# Install googletest
RUN git clone https://github.com/google/googletest
RUN cd googletest && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON .. && make && make install


# Install OpenCL
RUN git clone --recursive https://github.com/KhronosGroup/OpenCL-SDK.git
WORKDIR /tmp/OpenCL-SDK
RUN cmake -DBUILD_TESTING=OFF \
      -DBUILD_DOCS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DOPENCL_SDK_BUILD_SAMPLES=OFF \
      -DOPENCL_SDK_TEST_SAMPLES=OFF \
      -B ./build -S .
RUN cmake --build ./build --target install
RUN cp -r ./install/include/* /usr/local/include
RUN cp ./install/lib/libOpenCL.so /usr/local/lib/


# Set environment
ENV TVM_HOME="/workspace/opensource/tvm"
ENV TVM_CONFIGS_JSON_DIR="${TVM_HOME}/configs/host"
ENV TVM_LOG_DEBUG="ir/transform.cc=1,relay/ir/transform.cc=1"
ENV PYTHONPATH="${TVM_HOME}/python:${PYTHONPATH}"
ENV PATH="${PATH}:/usr/local/cuda/bin/:/usr/local/bin/ngc-cli"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/"


# Clean up
RUN rm -rf /tmp/* && apt autopurge -y && apt autoclean -y


EXPOSE 22
EXPOSE 8888
WORKDIR /home/george
CMD ["sleep", "infinity"]
