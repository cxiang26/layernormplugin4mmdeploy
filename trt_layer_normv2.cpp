#include<cuda.h>
#include<cuda_fp16.h>
#include<stdexcept>
#include "trt_serialize.hpp"
#include "trt_layer_normv2.hpp"
#include "trt_layer_normv2_kernel.hpp"
using namespace nvinfer1;

// #define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
// bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
//   if(code != cudaSuccess){
//     const char* err_name = cudaGetErrorName(code);
//     const char* err_message = cudaGetErrorString(code);
//     printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
//     return false;
//   }
//   return true;
// }

namespace mmdeploy {
namespace {
constexpr const char* PLUGIN_VERSION{"1"};
constexpr const char* PLUGIN_NAME{"TRTLayerNormalizationv2"};
}  // namespace

TRTLayerNormalizationv2::TRTLayerNormalizationv2(const std::string &name, float epsilon, std::size_t length,
    const std::vector<float> gamma, const std::vector<float> beta)
:    TRTPluginBase(name), epsilon_(epsilon), length_(length),  h_gamma(gamma), h_beta(beta){}

TRTLayerNormalizationv2::TRTLayerNormalizationv2(const std::string &name, const void *serialData, size_t serialLength)
:   TRTPluginBase(name)
{
    // WHERE_AM_I();
    // memcpy(&m_, buffer, sizeof(m_));
    // deserialize_value(&serialData, &serialLength, &mEpsilon);
    // LOG_INFO("get weight form buffer");
    // LOG_INFO("读取epsilon");
    memcpy(&epsilon_, serialData, sizeof(float));

    // LOG_INFO("读取length");
    reinterpret_cast<char const *&>(serialData) += sizeof(float);
    memcpy(&length_, serialData, sizeof(std::size_t));
      

    // LOG_INFO("读取gammma");
    reinterpret_cast<char const *&>(serialData) += sizeof(std::size_t);
    h_gamma.resize(length_);
    memcpy(h_gamma.data(), serialData, sizeof(float) * length_);

    // LOG_INFO("读取beta");
    reinterpret_cast<char const *&>(serialData) += sizeof(float) * length_;
    // float * p2 = static_cast<float *>(malloc(sizeof(float) * 256));
    h_beta.resize(length_);
    memcpy(h_beta.data(), serialData, sizeof(float) * length_);

    reinterpret_cast<char const *&>(serialData) -= sizeof(float) * 256;
    reinterpret_cast<char const *&>(serialData) -= sizeof(std::size_t);
    reinterpret_cast<char const *&>(serialData) -= sizeof(float);
    // LOG_INFO("epsilon is " + std::to_string(epsilon_));
    // LOG_INFO("length is " + std::to_string(length_));
    // LOG_INFO("gamma is " + std::to_string(h_gamma[0]));
    // LOG_INFO("beta is " + std::to_string(h_beta[0]));
}

TRTLayerNormalizationv2::~TRTLayerNormalizationv2(){}

size_t TRTLayerNormalizationv2::getSerializationSize() const noexcept
  {
      size_t length = sizeof(epsilon_) + sizeof(length_) + sizeof(float) * (
          h_gamma.size() + h_beta.size());
    //   LOG_INFO("the length of serialized data is " + std::to_string(length));
      return length;
  }

void TRTLayerNormalizationv2::serialize(void *buffer) const noexcept
  {
    // LOG_INFO("====== begin serialize ============");

    // 写入epsilon
	// LOG_INFO("epsilon is " + std::to_string(epsilon_));
    memcpy(buffer, &epsilon_, sizeof(epsilon_));

    // 写入length
	// LOG_INFO("length is " + std::to_string(length_));
    reinterpret_cast<char*&>(buffer) += sizeof(epsilon_);
    memcpy(buffer, &length_, sizeof(length_));

    // 写入gamma
	// LOG_INFO("gamma is " + std::to_string(h_gamma[0]));
    reinterpret_cast<char*&>(buffer) += sizeof(length_);
    memcpy(buffer, h_gamma.data(), sizeof(float) * h_gamma.size());

    // 写入beta
	// LOG_INFO("beta is " + std::to_string(h_beta[0]));
    reinterpret_cast<char*&>(buffer) += sizeof(float) * h_gamma.size();
    memcpy(buffer, h_beta.data(), sizeof(float) * h_beta.size());

    // 回退buffer指针
    reinterpret_cast<char*&>(buffer) -= sizeof(float) * h_gamma.size();
    reinterpret_cast<char*&>(buffer) -= sizeof(length_);
    reinterpret_cast<char*&>(buffer) -= sizeof(epsilon_);
  }

nvinfer1::IPluginV2DynamicExt *TRTLayerNormalizationv2::clone() const noexcept
  {
    // LOG_INFO("======== begin clone =========");
    // LOG_INFO("epsilon is " + std::to_string(epsilon_));
    // LOG_INFO("length is " + std::to_string(length_));
    // LOG_INFO("gamma is " + std::to_string(h_gamma[0]));
    // LOG_INFO("beta is " + std::to_string(h_beta[0]));
    return new TRTLayerNormalizationv2(mLayerName, epsilon_, length_, h_gamma, h_beta);
  }

int32_t TRTLayerNormalizationv2::getNbOutputs() const TRT_NOEXCEPT
{
    return 1;
}

nvinfer1::DimsExprs TRTLayerNormalizationv2::getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT
{
    return inputs[0];
}
   
bool TRTLayerNormalizationv2::supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) TRT_NOEXCEPT
{
    if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
    {
        return false;
    }

    bool res = false;
    switch (pos)
    {
    case 0:
        res = (inOut[pos].type == nvinfer1::DataType::kFLOAT) || (inOut[pos].type == nvinfer1::DataType::kHALF);
        break;
    case 1:
        res = inOut[pos].type == inOut[0].type;
        break;
    default: // should NOT be here
        break;
    }
    return res;
}

nvinfer1::DataType TRTLayerNormalizationv2::getOutputDataType(int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const TRT_NOEXCEPT
{
    return inputTypes[0];
}
 
void TRTLayerNormalizationv2::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs, const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) TRT_NOEXCEPT
{}

size_t TRTLayerNormalizationv2::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc *outputs, int32_t nbOutputs) const TRT_NOEXCEPT
{
    return 0;
}

void TRTLayerNormalizationv2::setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT
{
    mNamespace = pluginNamespace;
    return;
}

const char *TRTLayerNormalizationv2::getPluginNamespace() const TRT_NOEXCEPT
{
    return mNamespace.c_str();
}

const char *TRTLayerNormalizationv2::getPluginType() const TRT_NOEXCEPT
{
    return PLUGIN_NAME;
}
const char *TRTLayerNormalizationv2::getPluginVersion() const TRT_NOEXCEPT
{
    return PLUGIN_VERSION;
}

int32_t TRTLayerNormalizationv2::initialize() TRT_NOEXCEPT
{
    return 0;
}
void TRTLayerNormalizationv2::terminate() TRT_NOEXCEPT
{
    return;
}
void TRTLayerNormalizationv2::destroy() TRT_NOEXCEPT{}

int32_t TRTLayerNormalizationv2::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT
{
    int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    int nValuePerBlock = inputDesc[0].dims.d[2];
    if (inputDesc[0].type == DataType::kFLOAT)
    {
        // const auto* const gamma = static_cast<const float*>(d_gamma_.get());
        // const auto* const beta = static_cast<const float*>(d_beta_.get());
        // std::size_t nbytes = length_ * sizeof(float);
        // void * temp1 {nullptr};
        // void * temp2 {nullptr};
        // checkRuntime(cudaMallocAsync(&temp1, nbytes, stream));
        // checkRuntime(cudaMallocAsync(&temp2, nbytes, stream));
        // checkRuntime(
        //     cudaMemcpyAsync(
        //         temp1, h_gamma.data(), nbytes, cudaMemcpyHostToDevice, stream));
        // checkRuntime(
        //     cudaMemcpyAsync(
        //         temp2, h_beta.data(), nbytes, cudaMemcpyHostToDevice, stream));
        // float * gamma = static_cast<float *>(temp1);
        // float * beta = static_cast<float *>(temp2);

        const int VPT = 16 / sizeof(float);
        switch (nValuePerBlock)
        {
        case 256: { // 仅用于处理 nHiddenDimension 为 256 的情况SA
          const int TPB = 256 / VPT; 
        //   (LayerNormV2<float, 256, TPB, VPT>)<<<nBlock, TPB, 0, stream>>>(
        //       (float *)inputs[0], (float *)outputs[0],
        //       epsilon_, gamma, beta
        //   );
          (LayerNormV2Kernel<float, 256, TPB, VPT>)(
              (float *)inputs[0], (float *)outputs[0],
              epsilon_, (const float *)inputs[1], (const float *)inputs[2], nBlock, stream
          );
        //   LOG_INFO("调用核函数完成, FLOAT32类型, length 256");
          break;
        }
        case 768: { // 仅用于处理 nHiddenDimension 为 768 的情况
          const int TPB = 768 / VPT; 
        //   (LayerNormV2<float, 768, TPB, VPT>)<<<nBlock, TPB, 0, stream>>>(
        //       (float *)inputs[0], (float *)outputs[0],
        //        epsilon_, gamma, beta
        //   );
          (LayerNormV2Kernel<float, 768, TPB, VPT>)(
              (float *)inputs[0], (float *)outputs[0],
              epsilon_, (const float *)inputs[1], (const float *)inputs[2], nBlock, stream
          );
        //   LOG_INFO("调用核函数完成, FLOAT32类型, length 768");
          break;
        }
        case 32:
        //   (LayerNormV2<float, 32, 32, 1>)<<<nBlock, nValuePerBlock, 0, stream>>>(
        //       (float *)inputs[0], (float *)outputs[0],
        //       epsilon_, gamma, beta
        //   );
          (LayerNormV2Kernel<float, 32, 32, 1>)(
              (float *)inputs[0], (float *)outputs[0],
              epsilon_, (const float *)inputs[1], (const float *)inputs[2], nBlock, stream
          );
        //   LOG_INFO("调用核函数完成, FLOAT32类型, length 8");
          break;
        default: // shoulf NOT be here
            // LOG_ERROR("你输入的长度类型" + std::to_string(nValuePerBlock) + "暂不支持");
            // LOG_ERROR("当前仅支持长度768, 256, 8三种");
            break;
        }
    //   cudaDeviceSynchronize();
    //   cudaFree(gamma);
    //   cudaFree(beta);
    }
    else
    {
        // std::size_t nbytes = length_ * sizeof(half);
        // half * temp1 {nullptr};
        // half * temp2 {nullptr};
        // checkRuntime(cudaMallocAsync(&temp1, nbytes, stream));
        // checkRuntime(cudaMallocAsync(&temp2, nbytes, stream));
        // checkRuntime(
        //     cudaMemcpyAsync(
        //         temp1, h_gamma.data(), nbytes, cudaMemcpyHostToDevice, stream));
        // checkRuntime(
        //     cudaMemcpyAsync(
        //         temp2, h_beta.data(), nbytes, cudaMemcpyHostToDevice, stream));
        // half * gamma = static_cast<half *>(temp1);
        // half * beta = static_cast<half *>(temp2);
        const int VPT = 16 / sizeof(half);
        switch (nValuePerBlock)
        {
        case 256: { // 仅用于处理 nHiddenDimension 为 256 的情况
          const int TPB1 = 256 / VPT;
        //   (LayerNormV2<half, 256, TPB1, VPT>)<<<nBlock, TPB1, 0, stream>>>(
        //     (half *)inputs[0], (half *)outputs[0], epsilon_, gamma, beta
        //   );
          (LayerNormV2Kernel<half, 256, TPB1, VPT>)(
              (half *)inputs[0], (half *)outputs[0],
              epsilon_, (const half *)inputs[1], (const half *)inputs[2], nBlock, stream
          );
        //   LOG_INFO("调用核函数完成, FP16类型, length 256");
          break;
        }
        case 768: { // 仅用于处理 nHiddenDimension 为 768 的情况
          const int TPB2 = 768 / VPT;
        //   (LayerNormV2<half, 768, TPB2, VPT>)<<<nBlock, TPB2, 0, stream>>>(
        //       (half *)inputs[0], (half *)outputs[0], epsilon_, gamma, beta
        //   );
          (LayerNormV2Kernel<half, 768, TPB2, VPT>)(
              (half *)inputs[0], (half *)outputs[0],
              epsilon_, (const half *)inputs[1], (const half *)inputs[2], nBlock, stream
          );
        //   LOG_INFO("调用核函数完成, FP16类型, length 768");
          break;
        } 
        case 32: // 仅用于处理 nHiddenDimension 为 32 的情况
            // (LayerNormV2<half, 32, 32, 1>)<<<nBlock, nValuePerBlock, 0, stream>>>(
            //     (half *)inputs[0], (half *)outputs[0],
            //     epsilon_, gamma, beta
            // );
           (LayerNormV2Kernel<half, 32, 32, 1>)(
              (half *)inputs[0], (half *)outputs[0],
              epsilon_, (const half *)inputs[1], (const half *)inputs[2], nBlock, stream
           );
            // LOG_INFO("调用核函数完成, FP16类型, length 8");
            break;

        default: // shoulf NOT be here
            // LOG_ERROR("你输入的长度类型" + std::to_string(nValuePerBlock) + "暂不支持");
            // LOG_ERROR("当前仅支持长度768, 256, 8三种");
            break;
        }
    }
    return 0;   
}

TRTLayerNormalizationv2Creator::TRTLayerNormalizationv2Creator()
{
    mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("gamma", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("beta", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
	// LOG_INFO("\nmPluginAttributes.size=" + std::to_string(mPluginAttributes.size()));
}

TRTLayerNormalizationv2Creator::~TRTLayerNormalizationv2Creator(){}

const char *TRTLayerNormalizationv2Creator::getPluginName() const TRT_NOEXCEPT
{
    return PLUGIN_NAME;
}
const char *TRTLayerNormalizationv2Creator::getPluginVersion() const TRT_NOEXCEPT
{
    return PLUGIN_VERSION;
}
const nvinfer1::PluginFieldCollection *TRTLayerNormalizationv2Creator::getFieldNames() TRT_NOEXCEPT
{
    return &mFC;
}
nvinfer1::IPluginV2 *TRTLayerNormalizationv2Creator::createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT
{
    float epsilon {1.0e-5f};
    std::vector<float> gamma;
    std::vector<float> beta;
    std::size_t length = 0;;
    // LOG_INFO("============= create plugin  =============");
    // LOG_INFO("num of outputs is " + std::to_string(fc->nbFields));
    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
		// LOG_INFO("field name is " + field_name);
        if (field_name.compare("epsilon") == 0) {
          epsilon = *static_cast<const float *>(fc->fields[i].data);
        }

        if (field_name.compare("gamma") == 0) {
          const float * temp_gamma = static_cast<const float *>(fc->fields[i].data);
          length = fc->fields[i].length;
          gamma.resize(length);
          std::copy(temp_gamma, temp_gamma + length, gamma.begin());
        }

        if (field_name.compare("beta") == 0) {
          const float * temp_beta = static_cast<const float *>(fc->fields[i].data);
          length = fc->fields[i].length;
          beta.resize(length);
          std::copy(temp_beta, temp_beta + length, beta.begin());
        }
    }

    // LOG_INFO("epsilon is " + std::to_string(epsilon));
    // LOG_INFO("length is " + std::to_string(length));
    // LOG_INFO("gamma is " + std::to_string(gamma[0]));
    // LOG_INFO("beta is " + std::to_string(beta[0]));
    // LOG_INFO("====================================");
    return new TRTLayerNormalizationv2(name, epsilon, length, gamma, beta);
}
nvinfer1::IPluginV2 *TRTLayerNormalizationv2Creator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT
{
    return new TRTLayerNormalizationv2(name, serialData, serialLength);
}
void TRTLayerNormalizationv2Creator::setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT
{   
    mNamespace = pluginNamespace;
}
const char *TRTLayerNormalizationv2Creator::getPluginNamespace() const TRT_NOEXCEPT
{
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(TRTLayerNormalizationv2Creator);
}