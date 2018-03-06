FROM python:3.6

WORKDIR /usr/app/
ENTRYPOINT ["pip", "install", "requirements.txt"]

COPY ./requirements.txt .
