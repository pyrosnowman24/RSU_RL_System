FROM car2x/docker-veins-0.0.4

RUN apt-get upgrade && \
    apt-get --allow-releaseinfo-change update && \
    apt-get install -y  \
    git

RUN mkdir -p src 

WORKDIR /work/src

RUN git clone https://github.com/sommer/veins.git

WORKDIR /work/src/veins

R# UN ./configure
    
# RUN cd examples/veins --launchd -- ./run -u Cmdenv

# RUN head work/src/veins/examples/veins/results/General-\#0.sca
