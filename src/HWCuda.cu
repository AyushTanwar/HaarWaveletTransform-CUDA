#include "HWCuda.h"

__global__ void multDerechaGPU(float* const original_image, float* const changed_image,
	float* const mask, int rows, int cols, bool quantization_variable) {
	int row = (blockIdx.x * blockDim.x) + threadIdx.x;
	int col = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (col >= cols || row >= rows) {
		return;
	}

	int idx = row * cols + col;

	int r_i = threadIdx.x;  
	int c_i = threadIdx.y;  

	int rowStart = (blockIdx.x * blockDim.x);
	int colStart = (blockIdx.y * blockDim.y);

	float sum = 0;
	for (int k = 0; k < DIM_MASK; k++) {
		int index_mask = (r_i * DIM_MASK) + k;
		int index_imagen = ((rowStart + k) * cols) + (c_i + colStart);
		if (quantization_variable)
			sum = sum + mask[index_mask] * original_image[index_imagen];
		else
			sum = sum + mask[index_mask] * (original_image[index_imagen] * DELTA);
	}
	changed_image[idx] = sum;
}

__global__ void multIzquierdaGPU(float* const original_image, float* const changed_image,
	float* const mask, int rows, int cols, bool quantization_variable) {
	int row = (blockIdx.x * blockDim.x) + threadIdx.x;
	int col = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (col >= cols || row >= rows) {
		return;
	}

	int idx = row * cols + col;

	int rowStart = (blockIdx.x * blockDim.x);
	int colStart = (blockIdx.y * blockDim.y);

	int r = threadIdx.x;  /// renglon inicial del bloque 8 x 8
	int c = threadIdx.y;  /// col inicial del bloque 8 x 8

	float sum = 0;
	for (int k = 0; k < DIM_MASK; k++) {
		int index_mask = (k * DIM_MASK) + c;
		int index_imagen = ((rowStart + r) * rows) + (k + colStart);
		sum = sum + mask[index_mask] * original_image[index_imagen];
	}
	if (quantization_variable)
		changed_image[idx] = std::round(sum / DELTA);
	else
		changed_image[idx] = sum;
}

void HaarWaveletWrapper::MultDerechaCPU(float* const original_image, float* const changed_image,
	float* const mask, int rows, int cols, bool quantization_variable) {
	for (int row = 0; row < rows; row += DIM_MASK) {
		for (int col = 0; col < cols; col += DIM_MASK) {
			for (int r = 0; r < DIM_MASK; r++) {
				for (int c = 0; c < DIM_MASK; c++) {
					float sum = 0;
					for (int k = 0; k < DIM_MASK; k++) {
						int index_mask = (r * DIM_MASK) + k;
						int index_imagen = ((row + k) * cols) + (c + col);
						if (quantization_variable)
							sum = sum + mask[index_mask] * original_image[index_imagen];
						else
							sum = sum + mask[index_mask] * (original_image[index_imagen] * DELTA);
					}
					int idx = (row + r) * cols + (col + c);
					changed_image[idx] = sum;
				}
			}
		}
	}
}

void HaarWaveletWrapper::MultIzquierdaGPU(dim3 grid_size, dim3 blockSize, float* const original_image, float* const changed_image, float* const mask, int rows, int cols, bool quantization_variable)
{
	multIzquierdaGPU << <grid_size, blockSize >> >
		(original_image, changed_image, mask,
			rows, cols, quantization_variable);
}

void HaarWaveletWrapper::MultIzquierdaCPU(float* const original_image, float* const changed_image,
	float* const mask, int rows, int cols, bool quantization_variable) {
	for (int row = 0; row < rows; row += DIM_MASK) {
		for (int col = 0; col < cols; col += DIM_MASK) {
			for (int r = 0; r < DIM_MASK; r++) {
				for (int c = 0; c < DIM_MASK; c++) {
					float sum = 0;
					for (int k = 0; k < DIM_MASK; k++) {
						int index_mask = (k * DIM_MASK) + c;
						int index_imagen = ((row + r) * rows) + (k + col);
						sum = sum + mask[index_mask] * original_image[index_imagen];
					}
					int idx = (row + r) * cols + (col + c);
					if (quantization_variable)
						changed_image[idx] = round(sum / DELTA);
					else
						changed_image[idx] = sum;
				}
			}
		}
	}
}

void HaarWaveletWrapper::MultDerechaGPU(dim3 grid_size, dim3 blockSize, float* const original_image, float* const changed_image, float* const mask, int rows, int cols, bool quantization_variable)
{
	multDerechaGPU << <grid_size, blockSize >> >
		(original_image, changed_image, mask,
			rows, cols, quantization_variable);
}