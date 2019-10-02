FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

RUN conda install -y -c conda-forge opencv \
    && conda clean -ya

RUN pip uninstall -y pillow \
    && pip install pillow-simd \
    && pip install mlconfig mlflow scipy torchfile tqdm \
    && rm -rf ~/.cache/pip

WORKDIR /workspace
