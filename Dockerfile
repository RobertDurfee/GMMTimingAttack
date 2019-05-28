FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip

RUN pip3 install \
    numpy \
    matplotlib \
    pandas \
    scikit-learn

WORKDIR /workspace
