#ifndef HEADER_YOLOV5_API_
#define HEADER_YOLOV5_API_

#include <iostream>
#include <chrono>
#include <mutex>

#include "NvInfer.h"

#ifndef TRT_YOLOV5_COMMON_UTILS_H_
#define TRT_YOLOV5_COMMON_UTILS_H_
enum RUN_MODE
{
	USE_INT8,
	USE_FP16,
	USE_FP32
};
#endif  // TRT_YOLOV5_COMMON_UTILS_H_

struct yolov5_api
{
public:
	yolov5_api() {};
	~yolov5_api() {};


	virtual void initialization(
		RUN_MODE mode,
		int maxBatchSize,
		int deviceid,
		int _inSize_w,
		int _inSize_h,
		int _numClass) = 0;

	virtual void create_engine(
		std::string& wts_name,
		std::string& engine_name) = 0;

	virtual void load_engine(
		std::string& engine_name,
		int deviceid) = 0;

	virtual void doInference(float* input, float* output, int batchSize) = 0;
	// virtual void doInference(float* output, int batchSize) = 0;
	// virtual void* get_cuda_stream() = 0;
	// virtual void* get_input_buff() = 0;

	virtual int getOutputSize() = 0;

	virtual bool is_ready() const = 0;
};
#endif
