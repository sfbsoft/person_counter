FROM ubuntu:18.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -y update && \
    apt-get install -y libpq-dev build-essential python3-all-dev python3 python3-pip \
    libxml2-dev wget git cmake \
    libglib2.0-0 libsm6 libxrender1 libfontconfig1 \
    gnupg xserver-xorg x11-apps xorg-dev
RUN pip3 install -U pip
WORKDIR /app
COPY . /app
# RUN cd /app/src && pip install -r requirements.txt
RUN apt-get install -y jq
RUN wget https://github.com/soracom/soracom-cli/releases/download/v0.10.2/soracom_0.10.2_amd64.deb && \
    dpkg -i soracom_0.10.2_amd64.deb && \
    rm soracom_0.10.2_amd64.deb

CMD tail -f /dev/null 
