#include <assert.h>
#include <vector>
#include <iostream>
#include "yololayerv5.cuh"
#include "cuda_utils.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/cudacodec.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>


namespace Tn
{
    template<typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T>
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPluginV5::YoloLayerPluginV5(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel)
    {
        mClassCount = classCount;
        mYoloV5NetWidth = netWidth;
        mYoloV5NetHeight = netHeight;
        mMaxOutObject = maxOut;
        mYoloKernel = vYoloKernel;
        mKernelCount = vYoloKernel.size();

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float) * CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
    }
    YoloLayerPluginV5::~YoloLayerPluginV5()
    {
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaFree(mAnchor[ii]));
        }
        CUDA_CHECK(cudaFreeHost(mAnchor));
    }

    // create the plugin at runtime from a byte stream
    YoloLayerPluginV5::YoloLayerPluginV5(const void* data, size_t length)
    {
        using namespace Tn;
        const char* d = reinterpret_cast<const char*>(data), * a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        read(d, mYoloV5NetWidth);
        read(d, mYoloV5NetHeight);
        read(d, mMaxOutObject);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(mYoloKernel.data(), d, kernelSize);
        d += kernelSize;
        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float) * CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
        assert(d == a + length);
    }

    void YoloLayerPluginV5::serialize(void* buffer) const noexcept
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), * a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        write(d, mYoloV5NetWidth);
        write(d, mYoloV5NetHeight);
        write(d, mMaxOutObject);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(d, mYoloKernel.data(), kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }

    size_t YoloLayerPluginV5::getSerializationSize() const noexcept
    {
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(Yolo::YoloKernel) * mYoloKernel.size() + sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight) + sizeof(mMaxOutObject);
    }

    int YoloLayerPluginV5::initialize() noexcept
    {
        return 0;
    }

    Dims YoloLayerPluginV5::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
    {
        //output the result to channel
        int totalsize = mMaxOutObject * sizeof(Detection) / sizeof(float);

        return Dims3(totalsize + 1, 1, 1);
    }

    // Set plugin namespace
    void YoloLayerPluginV5::setPluginNamespace(const char* pluginNamespace) noexcept
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPluginV5::getPluginNamespace() const noexcept
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index 
    DataType YoloLayerPluginV5::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloLayerPluginV5::isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloLayerPluginV5::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
    {
        return false;
    }

    void YoloLayerPluginV5::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloLayerPluginV5::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
    {
    }

    // Detach the plugin object from its execution context.
    void YoloLayerPluginV5::detachFromContext() noexcept
    {
    }

    const char* YoloLayerPluginV5::getPluginType() const noexcept
    {
        return "YoloLayer_TRT";
    }

    const char* YoloLayerPluginV5::getPluginVersion() const noexcept
    {
        return "1";
    }

    void YoloLayerPluginV5::destroy() noexcept
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* YoloLayerPluginV5::clone() const noexcept
    {
        YoloLayerPluginV5* p = new YoloLayerPluginV5(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, mYoloKernel);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float LogistV5(float data) { return 1.0f / (1.0f + expf(-data)); };

    __global__ void CalDetection(const float* input, float* output, int noElements,
        const int netwidth, const int netheight, int maxoutobject, int yoloWidth, int yoloHeight, const float anchors[CHECK_COUNT * 2], int classes, int outputElem)
    {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid * bnIdx;
        int info_len_i = 5 + classes;
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

        for (int k = 0; k < 3; ++k) {
            float box_prob = LogistV5(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (box_prob < IGNORE_THRESH) continue;
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; ++i) {
                float p = LogistV5(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float* res_count = output + bnIdx * outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= maxoutobject) return;
            char* data = (char*)res_count + sizeof(float) + count * sizeof(Detection);
            Detection* det = (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            // pytorch:
            //  y = x[i].sigmoid()
            //  y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            //  y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh 
            //  X: (sigmoid(tx) + cx)/FeaturemapW *  netwidth 
            det->bbox[0] = (col - 0.5f + 2.0f * LogistV5(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netwidth / yoloWidth;
            det->bbox[1] = (row - 0.5f + 2.0f * LogistV5(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netheight / yoloHeight;

            // W: (Pw * e^tw) / FeaturemapW * netwidth  
            // v5: https://github.com/ultralytics/yolov5/issues/471
            det->bbox[2] = 2.0f * LogistV5(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
            det->bbox[2] = det->bbox[2] * det->bbox[2] * anchors[2 * k];
            det->bbox[3] = 2.0f * LogistV5(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
            det->bbox[3] = det->bbox[3] * det->bbox[3] * anchors[2 * k + 1];
            det->conf = box_prob * max_cls_prob;
            det->class_id = class_id;
        }
    }

    void YoloLayerPluginV5::forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize) noexcept
    {
        int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float);
        for (int idx = 0; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx * outputElem, 0, sizeof(float)));
        }
        int numElem = 0;
        for (unsigned int i = 0; i < mYoloKernel.size(); ++i)
        {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width * yolo.height * batchSize;
            if (numElem < mThreadCount)
                mThreadCount = numElem;

            //printf("Net: %d  %d \n", mYoloV5NetWidth, mYoloV5NetHeight);
            CalDetection << < (yolo.width * yolo.height * batchSize + mThreadCount - 1) / mThreadCount, mThreadCount >> >
                (inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float*)mAnchor[i], mClassCount, outputElem);
        }
    }


    int YoloLayerPluginV5::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
    {
        forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const noexcept
    {
        return "YoloLayer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const noexcept
    {
        return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames() noexcept
    {
        return &mFC;
    }

    IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
    {
#if(USE_YOLOV5_CF5_M)
        int class_count = -1;
        int input_w = -1;
        int input_h = -1;
        int max_output_object_count = -1;
        std::vector<Yolo::YoloKernel> yolo_kernels(3);

        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; i++) {
            if (strcmp(fields[i].name, "netdata") == 0) {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                int* tmp = (int*)(fields[i].data);
                class_count = tmp[0];
                input_w = tmp[1];
                input_h = tmp[2];
                max_output_object_count = tmp[3];
            }
            else if (strstr(fields[i].name, "yolodata") != NULL) {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                int* tmp = (int*)(fields[i].data);
                YoloKernel kernel;
                kernel.width = tmp[0];
                kernel.height = tmp[1];
                for (int j = 0; j < fields[i].length - 2; j++) {
                    kernel.anchors[j] = tmp[j + 2];
                }
                yolo_kernels[2 - (fields[i].name[8] - '1')] = kernel;
            }
        }
        assert(class_count && input_w && input_h && max_output_object_count);
        YoloLayerPluginV5* obj = new YoloLayerPluginV5(class_count, input_w, input_h, max_output_object_count, yolo_kernels);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
#endif

#if(USE_YOLOV5_CF5_M_CBLOCK_4STAGE || USE_YOLOV5_CF5_M_CBLOCK_4STAGE_INCIDENT)
        assert(fc->nbFields == 2);
        assert(strcmp(fc->fields[0].name, "netinfo") == 0);
        assert(strcmp(fc->fields[1].name, "kernels") == 0);
        int* p_netinfo = (int*)(fc->fields[0].data);
        int class_count = p_netinfo[0];
        int input_w = p_netinfo[1];
        int input_h = p_netinfo[2];
        int max_output_object_count = p_netinfo[3];
        std::vector<Yolo::YoloKernel> kernels(fc->fields[1].length);
        memcpy(&kernels[0], fc->fields[1].data, kernels.size() * sizeof(Yolo::YoloKernel));
        YoloLayerPluginV5* obj = new YoloLayerPluginV5(class_count, input_w, input_h, max_output_object_count, kernels);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
#endif
    }

    IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
    {
        // This object will be deleted when the network is destroyed, which will
        // call YoloLayerPlugin::destroy()
        YoloLayerPluginV5* obj = new YoloLayerPluginV5(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
}

