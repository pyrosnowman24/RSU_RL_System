FROM ubuntu:20.04

# Install libraries needed for imports
RUN apt-get upgrade && \
    apt-get update && \
    apt-get install -y  \
    git \
    nano \
    build-essentials \
    gcc \
    g++ \
    bison \
    flex \
    perl \
    tcl-dev \
    tk-dev \
    blt \
    libxm12-dev \
    zlib1g-dev \
    default-dev \
    doxygen \
    graphviz \
    libwebkitgtk-3.0-0 \
    openmpi-bin \
    libopenmpi-dev \
    libpcap-dev \
    autoconf \
    automake \
    libtool \
    libproj-dev \
    libgdal1-dev \
    libfox-1,6-dev \
    libgdal-dev \
    libcerces-c-dev \
    qt4-dev-tools \
    flex \
    perl \
    python \
    python3 \
    qt5-default \
    libqt5openg15-dev \
    tcl-dev \
    tk-dev \
    libxm12-dev \
    zlib1g-dev \
    default-jre \
    sumo \
    sumo-tools \
    sumo-doc

WORKDIR \app
RUN wget https://github.com/omnetpp/omnetpp/releases/download/omnetpp-5.7/omnetpp-5.7-linux-x86_64.tgz && \
    tar zxf omnetpp-5.7-linux-x86_64.tgz && \
    rm omnetpp-5.7-linux-x86_64.tgz && \
    cd omnetpp-5.7 && \
    export PATH=$PATH:~/app/omnetpp-5.7/bin && \
    ./configure && \
    make

RUN cd ~/app && wget https://veins.car2x.org/download/veins-5.2.zip && \
    unzip veins-5.2.zip && \
    rm veins-5.2.zip
