FROM continuumio/anaconda3:5.3.0
MAINTAINER "Gior Bernal"

RUN apt-get update -y && \
    apt-get install vim -y && \
    apt-get install procps -y && \
    apt-get install zip -y && \
    apt-get install unzip -y && \
    rm -rf /var/lib/apt/lists/*

RUN ["mkdir", "notebooks"]
RUN ["mkdir", "opt/dsbase"]

COPY src/main/dsbase/ /opt/dsbase/
COPY docker/conf/.jupyter /root/.jupyter
COPY docker/conf/.kaggle /root/.kaggle
COPY docker/run_jupyter.sh /
COPY docker/entrypoint.sh /

# Jupyter and Tensorboard ports
EXPOSE 8888 6006

# Store notebooks in this mounted directory
VOLUME /notebooks

CMD ["/entrypoint.sh"]
