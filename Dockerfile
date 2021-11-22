FROM python:3.8-buster

RUN pip3 install  pip setuptools wheel --upgrade
RUN pip3 install torch
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 8888
EXPOSE 8080
WORKDIR /app
CMD python3 -m jupyter notebook --ip 0.0.0.0 --allow-root