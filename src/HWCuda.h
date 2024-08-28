#pragma once

#include "../at.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DELTA 30
#define DIM_MASK 8
#define LONG_MASK 64

namespace HaarWaveletWrapper
{
	void MultIzquierdaGPU(dim3 grid_size, dim3 block_size, float* const original_image, float* const changed_image,
		float* const mask, int row, int cols, bool quantization_variable);

	void MultDerechaGPU(dim3 grid_size, dim3 block_size, float* const original_image, float* const changed_image,
		float* const mask, int row, int cols, bool quantization_variable);

	void MultDerechaCPU(float* const original_image, float* const changed_image,
		float* const mask, int row, int cols, bool quantization_variable);

	void MultIzquierdaCPU(float* const original_image, float* const changed_image,
		float* const mask, int row, int cols, bool quantization_variable);
};