FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

RUN conda install -y -c conda-forge opencv \
    && conda clean -ya

RUN pip uninstall -y pillow \
    && pip install pillow-simd \
    && rm -rf ~/.cache/pip

WORKDIR /workspace
