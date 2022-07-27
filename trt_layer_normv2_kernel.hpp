#ifndef TRT_LAYER_NORMv2_KERNEL_HPP
#define TRT_LAYER_NORMv2_KERNEL_HPP
#include <cublas_v2.h>
#include <cuda_runtime.h>

template<typename T, int n, int TPB, int VPT>
void LayerNormV2Kernel(T * input, T * output, float epsilon, const T * gamma, const T * beta, int nBlock, cudaStream_t stream);

#endif