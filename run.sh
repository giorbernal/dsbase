#!/bin/bash

docker run \
	--name=dsbase1 \
	-p 8888:8888 \
	-p 6006:6006 \
	-d \
	-e KAGGLE_USER='<myuser>' \
	-e KAGGLE_KEY='<mykey>' \
	--memory=6g \
	--memory-swap=-1 \
	-v $(pwd)/src/test:/notebooks \
	giorbernal/dsbase
