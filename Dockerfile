FROM ubuntu:18.04

RUN apt-get update; \
    apt-get install -y software-properties-common; \
    apt-get install python3.6 curl -y; \
    apt-get install python3-pip -y; \
    pip3 install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html; \
    pip3 install numpy gym box2d-py

# awscli is only necessary if this wil be run on aws services using the provided run_trl_job.sh script.
RUN pip3 install awscli

COPY ./ /usr/local/transfer_rl

WORKDIR /usr/local/transfer_rl
