FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

RUN apt-get -y update
RUN apt-get -y upgrade

# for setup time zone
RUN apt-get install -y tzdata
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN dpkg-reconfigure --frontend noninteractive tzdata

# RUN pip install -r requirements.txt