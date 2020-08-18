FROM continuumio/miniconda3:4.8.2

COPY environment.yaml /opt

RUN apt-get update && \
    apt-get install libgl1-mesa-glx -y

# Install chipsnet environment
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "source activate $(head -1 /opt/environment.yaml | cut -d' ' -f2)" > ~/.bashrc && \
    /opt/conda/bin/conda env create -f /opt/environment.yaml && \
    /opt/conda/bin/conda clean -afy
