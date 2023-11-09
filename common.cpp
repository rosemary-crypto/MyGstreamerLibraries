#include "pch.h"
#include "common.hpp"

cv::Rect get_rect(cv::Mat& img, float bbox[4], int w, int h, int cls) {
    int l, r, t, b;
    float r_w = w / (img.cols * 1.0);
    float r_h = h / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (h - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (h - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else {
        l = bbox[0] - bbox[2] / 2.f - (w - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (w - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    //assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");
    if (!input.is_open())
    {
        fprintf(stderr, "Unable to load weight file. please check if the .wts file path is right!!!!!!");
        exit(-1);
    }

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }
    std::cout << "Successfully loaded the weight file" << std::endl;

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    // silu = x * sigmoid
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

ILayer* focus(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname, int w, int h, int cls) {
    ISliceLayer* s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ inch, h / 2, w / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s2 = network->addSlice(input, Dims3{ 0, 1, 0 }, Dims3{ inch, h / 2, w / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s3 = network->addSlice(input, Dims3{ 0, 0, 1 }, Dims3{ inch, h / 2, w / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s4 = network->addSlice(input, Dims3{ 0, 1, 1 }, Dims3{ inch, h / 2, w / 2 }, Dims3{ 1, 2, 2 });
    ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
    return conv;
}

ILayer* bottleneck(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname) {
    auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

ILayer* bottleneckCSP(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = network->addConvolutionNd(input, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv2.weight"], emptywts);
    ITensor* y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolutionNd(*y1, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv3.weight"], emptywts);

    ITensor* inputTensors[] = { cv3->getOutput(0), cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);

    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), lname + ".bn", 1e-4);
    auto lr = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    auto cv4 = convBlock(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, lname + ".cv4");
    return cv4;
}

ILayer* C3(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv2");
    ITensor* y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }

    ITensor* inputTensors[] = { y1, cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);

    auto cv3 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv3");
    return cv3;
}

ILayer* SPP(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k1, k1 });
    pool1->setPaddingNd(DimsHW{ k1 / 2, k1 / 2 });
    pool1->setStrideNd(DimsHW{ 1, 1 });
    auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k2, k2 });
    pool2->setPaddingNd(DimsHW{ k2 / 2, k2 / 2 });
    pool2->setStrideNd(DimsHW{ 1, 1 });
    auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k3, k3 });
    pool3->setPaddingNd(DimsHW{ k3 / 2, k3 / 2 });
    pool3->setStrideNd(DimsHW{ 1, 1 });

    ITensor* inputTensors[] = { cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);

    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

std::vector<float> getAnchors(std::map<std::string, Weights>& weightMap)
{
    std::vector<float> anchors_yolo;
    Weights Yolo_Anchors = weightMap["model.24.anchor_grid"];
    assert(Yolo_Anchors.count == 18);
    int each_yololayer_anchorsnum = Yolo_Anchors.count / 3;
    const float* tempAnchors = (const float*)(Yolo_Anchors.values);
    for (int i = 0; i < Yolo_Anchors.count; i++)
    {
        if (i < each_yololayer_anchorsnum)
        {
            anchors_yolo.push_back(const_cast<float*>(tempAnchors)[i]);
        }
        if ((i >= each_yololayer_anchorsnum) && (i < (2 * each_yololayer_anchorsnum)))
        {
            anchors_yolo.push_back(const_cast<float*>(tempAnchors)[i]);
        }
        if (i >= (2 * each_yololayer_anchorsnum))
        {
            anchors_yolo.push_back(const_cast<float*>(tempAnchors)[i]);
        }
    }
    return anchors_yolo;
}

IPluginV2Layer* addYoLoLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, IConvolutionLayer* det0, IConvolutionLayer* det1, IConvolutionLayer* det2, int w, int h, int cls)
{
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    std::vector<float> anchors_yolo = getAnchors(weightMap);
    PluginField pluginMultidata[4];
    int NetData[4];
    NetData[0] = cls;
    NetData[1] = w;
    NetData[2] = h;
    NetData[3] = Yolo::MAX_OUTPUT_BBOX_COUNT;
    pluginMultidata[0].data = NetData;
    pluginMultidata[0].length = 3;
    pluginMultidata[0].name = "netdata";
    pluginMultidata[0].type = PluginFieldType::kFLOAT32;
    int scale[3] = { 8, 16, 32 };
    int plugindata[3][8];
    std::string names[3];
    for (int k = 1; k < 4; k++)
    {
        plugindata[k - 1][0] = w / scale[k - 1];
        plugindata[k - 1][1] = h / scale[k - 1];
        for (int i = 2; i < 8; i++)
        {
            plugindata[k - 1][i] = int(anchors_yolo[(k - 1) * 6 + i - 2]);
        }
        pluginMultidata[k].data = plugindata[k - 1];
        pluginMultidata[k].length = 8;
        names[k - 1] = "yolodata" + std::to_string(k);
        pluginMultidata[k].name = names[k - 1].c_str();
        pluginMultidata[k].type = PluginFieldType::kFLOAT32;
    }
    PluginFieldCollection pluginData;
    pluginData.nbFields = 4;
    pluginData.fields = pluginMultidata;
    IPluginV2* pluginObj = creator->createPlugin("yololayer", &pluginData);
    ITensor* inputTensors_yolo[] = { det2->getOutput(0), det1->getOutput(0), det0->getOutput(0) };
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
    return yolo;
}

std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) {
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"];
    int anchor_len = Yolo::CHECK_COUNT * 2;
    for (int i = 0; i < wts.count / anchor_len; i++) {
        auto* p = (const float*)wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}

IPluginV2Layer* addYoLoLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets, int w, int h, int cls, bool is_phrd) {

#if(USE_YOLOV5_CF5_M_CBLOCK_4STAGE_INCIDENT || USE_YOLOV5_CF5_X_CBLOCK_4STAGE_INCIDENT || USE_YOLOV5_CF5_M_CBLOCK_4STAGE_PHRD)
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, lname);
    PluginField plugin_fields[2];
    int netinfo[4] = { cls, w, h, Yolo::MAX_OUTPUT_BBOX_COUNT };
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    // int scale = 8;
    // int scale = 16;
    int scale[4] = { 0 };
    if (is_phrd)
    {
        scale[0] = 8;
        scale[1] = 16;
        scale[2] = 32;
        scale[3] = 64;
    }
    else
    {
        scale[0] = 4;
        scale[1] = 8;
        scale[2] = 16;
        scale[3] = 32;
    }

    std::vector<Yolo::YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        Yolo::YoloKernel kernel;
        kernel.width = w / scale[i];
        kernel.height = h / scale[i];
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        // scale *= 2;
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2* plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<ITensor*> input_tensors;
    for (auto det : dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;
#else
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, lname);
    PluginField plugin_fields[2];
    int netinfo[4] = { Yolo::CLASS_NUM, Yolo::INPUT_W, Yolo::INPUT_H, Yolo::MAX_OUTPUT_BBOX_COUNT };
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    // int scale = 8;
    // int scale = 16;
    int scale[4] = { 16, 16, 32, 64 };
    std::vector<Yolo::YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        Yolo::YoloKernel kernel;
        kernel.width = Yolo::INPUT_W / scale[i];
        kernel.height = Yolo::INPUT_H / scale[i];
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        // scale *= 2;
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2* plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<ITensor*> input_tensors;
    for (auto det : dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;
#endif
}

ILayer* compressorblock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname, int w, int h, int cls) {

    // std::cout<<"======== "<<inch<<std::endl;
    auto pool1 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{ 2, 2 });
    // auto pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 2, 2 });
    pool1->setPaddingNd(DimsHW{ 0 , 0 });
    pool1->setStrideNd(DimsHW{ 2, 2 });
    // std::cout<<"======== 1 "<<inch<<std::endl;

    auto conv1 = convBlock(network, weightMap, input, 4, 3, 2, 1, lname + ".cbr1");

    ITensor* inputTensors1[] = { conv1->getOutput(0), pool1->getOutput(0) };

    // std::cout<<"======== 3 "<<inch<<std::endl;

    auto cat1 = network->addConcatenation(inputTensors1, 2);

    // std::cout<<"======== 4 "<<inch<<std::endl;

    auto conv2 = convBlock(network, weightMap, *cat1->getOutput(0), 8, 3, 1, 1, lname + ".cbr3");

    std::cout << "======== 5 " << conv2->getOutput(0)->getDimensions().d[0] << " " << conv2->getOutput(0)->getDimensions().d[1] << " " << conv2->getOutput(0)->getDimensions().d[2] << std::endl;

    ISliceLayer* s1 = network->addSlice(*conv2->getOutput(0), Dims3{ 0, 0, 0 }, Dims3{ 8, h / 4, w / 4 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s2 = network->addSlice(*conv2->getOutput(0), Dims3{ 0, 1, 0 }, Dims3{ 8, h / 4, w / 4 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s3 = network->addSlice(*conv2->getOutput(0), Dims3{ 0, 0, 1 }, Dims3{ 8, h / 4, w / 4 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s4 = network->addSlice(*conv2->getOutput(0), Dims3{ 0, 1, 1 }, Dims3{ 8, h / 4, w / 4 }, Dims3{ 1, 2, 2 });

    // std::cout<<"======== 7 "<<inch<<std::endl;

    ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);

    std::cout << "======== 8 " << cat->getOutput(0)->getDimensions().d[0] << " " << cat->getOutput(0)->getDimensions().d[1] << " " << cat->getOutput(0)->getDimensions().d[2] << std::endl;

    auto conv = convBlock(network, weightMap, *cat->getOutput(0), 48, ksize, 1, 1, lname + ".conv");

    std::cout << "======== 9 " << conv->getOutput(0)->getDimensions().d[0] << " " << conv->getOutput(0)->getDimensions().d[1] << " " << conv->getOutput(0)->getDimensions().d[2] << std::endl;
    // 
        // ITensor* inputTensors1[] = {conv1->getOutput(0), conv->getOutput(0)};

        // std::cout<<"======== 3 "<<inch<<std::endl;

        // auto cat1 = network->addConcatenation(inputTensors1, 2);


        // auto conv2 = convBlock(network, weightMap, *cat1->getOutput(0), outch, 3, 2, 1, lname + ".cbr2");

    return conv;
#if(0)
    auto pool1 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{ 2, 2 });

    auto conv1 = convBlock(network, weightMap, input, 4, 3, 2, 1, lname + ".cbr1");

    ITensor* inputTensors1[] = { conv1->getOutput(0), pool1->getOutput(0) };

    auto cat1 = network->addConcatenation(inputTensors1, 2);

    auto conv2 = convBlock(network, weightMap, *cat1->getOutput(0), 8, 3, 1, 1, lname + ".cbr3");

    ISliceLayer* s1 = network->addSlice(*conv2->getOutput(0), Dims3{ 0, 0, 0 }, Dims3{ 8, h / 4, w / 4 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s2 = network->addSlice(*conv2->getOutput(0), Dims3{ 0, 1, 0 }, Dims3{ 8, h / 4, w / 4 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s3 = network->addSlice(*conv2->getOutput(0), Dims3{ 0, 0, 1 }, Dims3{ 8, h / 4, w / 4 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s4 = network->addSlice(*conv2->getOutput(0), Dims3{ 0, 1, 1 }, Dims3{ 8, h / 4, w / 4 }, Dims3{ 1, 2, 2 });

    ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);

    auto conv = convBlock(network, weightMap, *cat->getOutput(0), 48, ksize, 1, 1, lname + ".conv");

    return conv;
#endif
}

