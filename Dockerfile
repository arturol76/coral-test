FROM arturol76/phusion-baseimage:0.11
LABEL maintainer="arturol76"

RUN apt-get -y install python3-pip \
	&& pip3 install --upgrade pip

#deploy sample python app
COPY ./app /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000

#refer to https://github.com/phusion/baseimage-docker#docker_single_process
RUN mkdir /etc/service/app
COPY ./app/myapp.run /etc/service/app/run
RUN chmod +x /etc/service/app/run

