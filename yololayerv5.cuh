#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include "NvInfer.h"
#include "pch.h"

#define USE_YOLOV5_CF4_X                                        0
#define USE_YOLOV5_CF5_M                                        0
#define USE_YOLOV5_CF5_M_CBLOCK_4STAGE_INCIDENT                 1
#define USE_YOLOV5_CF5_M_CBLOCK_4STAGE                          0
#define USE_YOLOV5_CF5_X_CBLOCK_4STAGE_INCIDENT                 1
#define USE_YOLOV5_CF5_M_CBLOCK_4STAGE_PHRD                     1

#ifndef STRUCT_V5_BBOX
#define STRUCT_V5_BBOX
static constexpr int LOCATIONS = 4;
struct alignas(float) Detection {
    //center_x center_y w h
    float bbox[LOCATIONS];
    float conf;  // bbox_conf * cls_conf
    float class_id;
};
#endif

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
}

namespace nvinfer1
{
    class YoloLayerPluginV5 : public IPluginV2IOExt
    {
    public:
        YoloLayerPluginV5(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
        YoloLayerPluginV5(const void* data, size_t length);
        ~YoloLayerPluginV5();

        int getNbOutputs() const noexcept override
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

        int initialize() noexcept override;

        virtual void terminate() noexcept override
        {
        }

        virtual size_t getWorkspaceSize(int maxBatchSize) const noexcept override
        {
            return 0;
        }

        virtual int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

        virtual size_t getSerializationSize() const noexcept override;

        virtual void serialize(void* buffer) const noexcept override;

        bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept override
        {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char* getPluginType() const noexcept override;

        const char* getPluginVersion() const noexcept override;

        void destroy() noexcept override;

        IPluginV2IOExt* clone() const noexcept override;

        void setPluginNamespace(const char* pluginNamespace) noexcept override;

        const char* getPluginNamespace() const noexcept override;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

        bool isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override;

        bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

        void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

        void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept override;

        void detachFromContext() noexcept override;

    private:
        void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize = 1) noexcept;
        int mThreadCount = 256;
        const char* mPluginNamespace;
        int mKernelCount;
        int mClassCount;
        int mYoloV5NetWidth;
        int mYoloV5NetHeight;
        int mMaxOutObject;
        std::vector<Yolo::YoloKernel> mYoloKernel;
        void** mAnchor;
    };

    class YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        AsciiChar const* getPluginName() const noexcept override;

        AsciiChar const* getPluginVersion() const noexcept override;

        const PluginFieldCollection* getFieldNames() noexcept override;

        IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

        IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

        void setPluginNamespace(AsciiChar const* libNamespace) noexcept override
        {
            mNamespace = libNamespace;
        }

        AsciiChar const* getPluginNamespace() const noexcept override
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif 
