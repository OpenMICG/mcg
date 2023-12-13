conda create -n mcg python=3.8
conda activate mcg

sudo apt update
sudo apt install lsof

spacy download en_core_web_sm

pip install -r requirements.txt

# use the faster pillow-simd instead of the original pillow
# https://github.com/uploadcare/pillow-simd
pip uninstall pillow && \
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

git clone https://github.com/NVIDIA/apex.git &&\
    cd apex &&\
    pip install -v --no-cache-dir . &&\
    rm -rf ../apex

# horovod
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod
sudo ldconfig
