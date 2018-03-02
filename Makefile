
IMAGE ?= mlmagic
TAG ?= latest

all: jupyterlab

.PHONY: build
build:
	docker build -t ${IMAGE}:${TAG} .

.PHONY: oldrun
oldrun: build
	docker run --rm -it \
	-v $(shell pwd):/usr/app/src \
	-p 8080:8080 \
	${IMAGE}:${TAG} bash

.PHONY: run
run:
	docker run --rm -it \
	-v $(shell pwd):/home/jovyan/work/ \
	-p 8888:8888 \
	jupyter/scipy-notebook \
	start-notebook.sh

.PHONY: jupyterlab
jupyterlab: 
	docker run --rm -it \
	-v $(shell pwd):/home/jovyan/work/ \
	-p 8888:8888 \
	jupyter/datascience-notebook \
	start.sh jupyter lab