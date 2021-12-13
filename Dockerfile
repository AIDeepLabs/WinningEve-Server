FROM python:3.8-buster

RUN pip3 install  pip setuptools wheel --upgrade
RUN pip3 install torch
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN apt update
RUN apt install -y libgl1
# EXPOSE 8888
EXPOSE 9000
WORKDIR /app
CMD python3 server.py