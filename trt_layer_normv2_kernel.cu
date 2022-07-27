#include <cub/cub.cuh>
#include "trt_layer_normv2_kernel.hpp"
#include "trt_plugin_helper.hpp"
#include "common_cuda_helper.hpp"

template <int VPT>
struct BytesToType;
// 
template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};


template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;
// 
    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}


__device__ inline float2 operator + (const  float2 &a, const float2 & b) {
  float2 out{0.0f, 0.0f};
  // printf("a.x %f, b.x %f\t a.y %f, b.y %f\n", a.x, b.x, a.y, b.y);
  out.x = a.x + b.x;
  out.y = a.y + b.y;
  return out;
}


template<typename T, int n, int TPB, int VPT>
__global__ void LayerNormV2(
    T * input, T * output, float epsilon, const T * gamma, const T * beta) {
  /*
  cub v2版，将输出结果用float储存，无论中间是否为half数据还是float32数据
  使用cub内置的内存管理进行数据拷贝，减少数据拷贝时间。
  params T: 数据类型，默认为fp32或者fp16
  param n: 数据长度，对于三维输入变量，一般为最后一个维度的值
  param VPT: 为16除以数据类型的个数，16为数据对齐的最小单位, 展开计算速度会快一些
  param TPB: 数据束个数，如果n < 32, 则取32；否则取 n / VPT的值
  params input: 输入数据
  params output: 最终输出结果
  params epsilon: 除标准差的时候加的一个小系数，防止分母为零
  params gamma: 用于LayerNorm后相乘该对象
  params beta: 用于LayerNorm后相加该对象
  */
  const int idx = threadIdx.x * VPT + blockIdx.x * n;
  const int tx = threadIdx.x * VPT;
  // 准备本地数据
  T local_x[VPT], local_gamma[VPT], local_beta[VPT];
  // 复制一个束的数据到local相关数据中，加快速度
  copy<sizeof(T) * VPT>(&input[idx], local_x);
  // 计算长度的倒数，用于后续求均值用(注意这里还是用float，中间精度float更好)
  const float r_length = float(1) / float(n);
  // 储存均值和方差，同时完成
  float2 local_float2 {0.f, 0.f};

#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const float temp = r_length * (float)local_x[i];
    local_float2.x += temp;
    local_float2.y += temp * (float)local_x[i];
  }
  // 开始拷贝gamma与beta,注意，gamma与beta都是一维变量，只与tx有关，与idx无关
  copy<sizeof(T) * VPT>(&gamma[tx], local_gamma);
  copy<sizeof(T) * VPT>(&beta[tx], local_beta);
  // 利用blockReduce计算所有线程的均值和方差，均值与方差可以同时完成计算
  using BlockReduce = cub::BlockReduce<float2, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float mu; // 均值
  __shared__ float rsigma; // 1/标准差
  const float2 sum2 = BlockReduce(temp_storage).Reduce(local_float2, cub::Sum());

  // 获取最终结果
  if (threadIdx.x == 0) {

    // printf("idx %d tx %d sum2 x %d, y %d\n", idx, tx, sum2.x, sum2.y);
    mu = sum2.x;
    // 注意这里应该还缺一个epsilon * sqrt(sum2.y - mu * mu)，简化计算，所以忽略了
    // rsigma = 1 / (sqrt(sum2.y - mu * mu) + epsilon);
    rsigma = rsqrt(sum2.y - mu * mu + epsilon * epsilon);
  }
  __syncthreads();

  // 展开循环体 - 计算最终的LayerNorm的值
#pragma unroll
  // printf("idx %d tx %d   mu %f gamma %f beta %f\n", idx, tx, mu, local_gamma[0], local_beta[0]);
  for (int i = 0; i < VPT; ++i) {
    local_x[i] = (float)local_gamma[i] * (
      (float)local_x[i] - mu
    )  * rsigma + (float)local_beta[i];
  }
  // 将对应数据拷贝回output
  copy<sizeof(T) * VPT>(local_x, &output[idx]);
}

template<typename T, int n, int TPB, int VPT>
void LayerNormV2Kernel(T * input, T * output, float epsilon, const T * gamma, const T * beta, int nBlock, cudaStream_t stream)
{
    // LayerNormV2(input, output, epsilon, gamma, beta);
    (LayerNormV2<T, 256, TPB, VPT>)<<<nBlock, TPB, 0, stream>>>((T *)input, (T *)output, epsilon, (const T *)gamma, (const T *)beta);
}