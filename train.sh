set -ex

export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export OMP_NUM_THREADS=1


python ma_dreamerv3.py 