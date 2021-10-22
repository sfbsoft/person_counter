FROM ubuntu:18.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -y update && \
    apt-get install -y libpq-dev build-essential python3-all-dev python3 python3-pip \
    libxml2-dev wget git cmake curl \
    libglib2.0-0 libsm6 libxrender1 libfontconfig1 \
    gnupg xserver-xorg x11-apps xorg-dev
RUN pip3 install -U pip
WORKDIR /app
COPY . /app

# Soracom CLI
RUN apt-get install -y jq
RUN wget https://github.com/soracom/soracom-cli/releases/download/v0.10.2/soracom_0.10.2_amd64.deb && \
    dpkg -i soracom_0.10.2_amd64.deb && \
    rm soracom_0.10.2_amd64.deb

# Tensorflow Lite
RUN pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl

# Coral dependencies
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get -y update

# Standard
RUN apt-get -y install libedgetpu1-std

# Maximum
# RUN apt-get install libedgetpu1-max

# Install requirements
RUN pip3 install -r /app/requirements.txt

CMD tail -f /dev/null 
