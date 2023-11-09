#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "yololayerv5.cuh"
using namespace nvinfer1;

cv::Rect get_rect(cv::Mat& img, float bbox[4], int w, int h, int cls);

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file);

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);
ILayer* convBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname);
ILayer* focus(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname, int w, int h, int cls);
ILayer* bottleneck(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname);
ILayer* bottleneckCSP(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname);
ILayer* C3(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname);
ILayer* SPP(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string lname);

std::vector<float> getAnchors(std::map<std::string, Weights>& weightMap);
std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname);

IPluginV2Layer* addYoLoLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, IConvolutionLayer* det0, IConvolutionLayer* det1, IConvolutionLayer* det2, int w, int h, int cls);
IPluginV2Layer* addYoLoLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets, int w, int h, int cls, bool is_phrd = false);

ILayer* compressorblock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname, int w, int h, int cls);

#endif