#ifndef TRT_LAYER_NORMALIZATIONv2_HPP
#define TRT_LAYER_NORMALIZATIONv2_HPP

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <memory>

#include "trt_plugin_base.hpp"


namespace mmdeploy{
class TRTLayerNormalizationv2 final : public TRTPluginBase
{
private:
    std::vector<float> h_gamma;
    std::vector<float> h_beta;
    float epsilon_;
    std::size_t length_;

public:
    TRTLayerNormalizationv2(const std::string &name, float epsilon, std::size_t length,
    const std::vector<float> gamma, const std::vector<float> beta);
    TRTLayerNormalizationv2(const std::string &name, const void *buffer, size_t length);
    TRTLayerNormalizationv2() = delete;
    ~TRTLayerNormalizationv2();

    // Method inherited from IPluginV2
    const char *getPluginType() const TRT_NOEXCEPT override;
    const char *getPluginVersion() const TRT_NOEXCEPT override;
    int32_t     getNbOutputs() const TRT_NOEXCEPT override;
    int32_t     initialize() TRT_NOEXCEPT override;
    void        terminate() TRT_NOEXCEPT override;
    size_t      getSerializationSize() const TRT_NOEXCEPT override;
    void        serialize(void *buffer) const TRT_NOEXCEPT override;
    void        destroy() TRT_NOEXCEPT override;
    void        setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override;
    const char *getPluginNamespace() const TRT_NOEXCEPT override;

    // Method inherited from IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const TRT_NOEXCEPT override;
    void     attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, nvinfer1::IGpuAllocator *gpuAllocator) TRT_NOEXCEPT override;
    void     detachFromContext() TRT_NOEXCEPT override;

    //Method inherited from IPluginV2DynamicExt
    nvinfer1::IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override;
    nvinfer1::DimsExprs            getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT override;
    bool                 supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) TRT_NOEXCEPT override;
    void                 configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs, const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) TRT_NOEXCEPT override;
    size_t               getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc *outputs, int32_t nbOutputs) const TRT_NOEXCEPT override;
    int32_t              enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override;

};

class TRTLayerNormalizationv2Creator : public TRTPluginCreatorBase
{
private:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;

public:
    TRTLayerNormalizationv2Creator();
    ~TRTLayerNormalizationv2Creator();
    const char *                           getPluginName() const TRT_NOEXCEPT override;
    const char *                           getPluginVersion() const TRT_NOEXCEPT override;
    const nvinfer1::PluginFieldCollection  *getFieldNames() TRT_NOEXCEPT override;
    nvinfer1::IPluginV2 *                  createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT override;
    nvinfer1::IPluginV2 *                  deserializePlugin(const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT override;
    void                                   setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override;
    const char *                           getPluginNamespace() const TRT_NOEXCEPT override;
   
};
}

#endif