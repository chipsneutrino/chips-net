FROM continuumio/miniconda3:4.7.12-alpine
LABEL name=chips-cvn

COPY environment.yml /opt

# Install chips-cvn environment
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "source activate $(head -1 /opt/environment.yml | cut -d' ' -f2)" > ~/.bashrc && \
    /opt/conda/bin/conda env create -f /opt/environment.yml && \
    /opt/conda/bin/conda clean -afy
    