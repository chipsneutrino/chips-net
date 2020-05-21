FROM continuumio/miniconda3:latest

COPY environment.yaml /opt

# Install chipsnet environment
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "source activate $(head -1 /opt/environment.yaml | cut -d' ' -f2)" > ~/.bashrc && \
    /opt/conda/bin/conda env create -f /opt/environment.yaml && \
    /opt/conda/bin/conda clean -afy
