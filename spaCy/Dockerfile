FROM continuumio/miniconda3

#Ensures bash will be default
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install some packages via conda
RUN conda create -n par && \
	source activate par && \
	conda install spacy && \
	conda install jupyter

# jupyter notebook --ip=0.0.0.0  --allow-root --port 8000
