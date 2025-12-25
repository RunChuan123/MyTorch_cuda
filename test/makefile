.PHONY: t1

t1: t1.cu /root/autodl-tmp/MiniTorch/src/ops_ori/gemm.cu
    nvcc -Xptxas -v t1.cu /root/autodl-tmp/MiniTorch/src/ops_ori/gemm.cu -arch=sm_75 -o t1