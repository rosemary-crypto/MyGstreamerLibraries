#pragma once

#include "yolov5_base.h"
#include "logging.h"
#include "my_macros.h"

extern "C" {
	struct yolov5phrd_trt : yolov5_api
	{
	public:
		MY_LIBRARY_API yolov5phrd_trt();
		MY_LIBRARY_API ~yolov5phrd_trt();

		MY_LIBRARY_API void initialization(
			RUN_MODE mode,
			int maxBatchSize,
			int deviceid,
			int _inSize_w,
			int _inSize_h,
			int _numClass) override;

		MY_LIBRARY_API void create_engine(
			std::string& wts_name,
			std::string& engine_name) override;

		MY_LIBRARY_API void load_engine(
			std::string& engine_name,
			int deviceid) override;

		MY_LIBRARY_API void doInference(float* input, float* output, int batchSize) override;

		MY_LIBRARY_API int getOutputSize() override;

		MY_LIBRARY_API bool is_ready() const;

	private:
		struct implement;
		implement* impl;
	};

} //extern "C"