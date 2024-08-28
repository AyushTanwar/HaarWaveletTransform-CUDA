
#include "HWIC.h"
#include <fstream>
#include <iostream>


using namespace cv;
using namespace std;

// quality-metric
namespace qm {
#define constant_a (float) (0.01 * 255 * 0.01  * 255)
#define constant_b (float) (0.03 * 255 * 0.03  * 255)


	
	double summation(Mat matrix, int s, int f, int block_size)
	{
		double rv = 0;

		Mat m_tmp = matrix(Range(s, s + block_size), Range(f, f + block_size));
		Mat m_squared(block_size, block_size, CV_64F);

		multiply(m_tmp, m_tmp, m_squared);

		// E(x)
		double mean1 = mean(m_tmp)[0];
		// E(x²)
		double mean2 = mean(m_squared)[0];

		rv = sqrt(mean2 - mean1 * mean1);

		return rv;
	}

	// Covariance
	double covariance(Mat a, Mat b, int s, int f, int block_size)
	{
		Mat c = Mat::zeros(block_size, block_size, a.depth());
		Mat a_tmp = a(Range(s, s + block_size), Range(f, f + block_size));
		Mat b_tmp = b(Range(s, s + block_size), Range(f, f + block_size));

		multiply(a_tmp, b_tmp, c);

		double avg_oc = mean(c)[0]; // E(XY)
		double mean_comp = mean(a_tmp)[0]; // E(X)
		double mean_orig = mean(b_tmp)[0]; // E(Y)

		double rv = avg_oc - mean_orig * mean_comp; // E(XY) - E(X)E(Y)

		return rv;
	}

	// Mean squared error
	double mean_square_error(Mat picture_1, Mat picture_2)
	{
		int i, j;
		double rv = 0;
		int breadth = picture_1.rows;
		int length = picture_1.cols;

		for (i = 0; i < breadth; i++)
			for (j = 0; j < length; j++)
				rv = rv + (picture_1.at<double>(i, j) - picture_2.at<double>(i, j)) * (picture_1.at<double>(i, j) - picture_2.at<double>(i, j));

		rv = rv/(breadth * length);

		return rv;
	}

	double peak_signal_noise_ratio(Mat original, Mat compressed, int block_size)
	{
		int D = 255;
		return (10 * log10((D * D) / mean_square_error(original, compressed)));
	}

	
	double structural_similarity_index_measure(Mat original, Mat compressed, int block_size)
	{
		double rv = 0;

		int no_of_block_breadth = original.rows / block_size;
		int no_of_block_length = original.cols / block_size;

		for (int k = 0; k < no_of_block_breadth; k++)
		{
			for (int l = 0; l < no_of_block_length; l++)
			{
				int m = k * block_size;
				int n = l * block_size;

				double mean_orig = mean(original(Range(k, k + block_size), Range(l, l + block_size)))[0];
				double mean_comp = mean(compressed(Range(k, k + block_size), Range(l, l + block_size)))[0];
				double sum_orig = summation(original, m, n, block_size);
				double sum_comp = summation(compressed, m, n, block_size);
				double mean_oc = covariance(original, compressed, m, n, block_size);

				rv = rv + ((2 * mean_orig * mean_comp + constant_a) * (2 * mean_oc + constant_b)) / ((mean_orig * mean_orig + mean_comp * mean_comp + constant_a) * (sum_orig * sum_orig + sum_comp * sum_comp + constant_b));
			}
			
		}
		rv = rv/(no_of_block_breadth * no_of_block_length);

		return rv;
	}
}

Mat zigZagger(Mat matrix_image) {
	Mat array_image = Mat::zeros(1, matrix_image.cols * matrix_image.rows, CV_32F);

	int array_index_image = 0;

	int x_ind = 0;
	int y_ind = 0;

	while (array_index_image < (matrix_image.cols * matrix_image.rows)) {
		if (x_ind < matrix_image.cols - 1) {
			array_image.at<float>(0, array_index_image) = matrix_image.at<float>(x_ind, y_ind);
			array_index_image++;
			x_ind = x_ind + 1;

			while (x_ind > 0) {
				array_image.at<float>(0, array_index_image) = matrix_image.at<float>(x_ind, y_ind);
				array_index_image++;
				x_ind = x_ind - 1;
				y_ind = y_ind + 1;
			}
		}
		else if (x_ind == matrix_image.cols - 1) {
			array_image.at<float>(0, array_index_image) = matrix_image.at<float>(x_ind, y_ind);
			array_index_image++;
			y_ind = y_ind + 1;

			while (y_ind < matrix_image.rows - 1) {
				array_image.at<float>(0, array_index_image) = matrix_image.at<float>(x_ind, y_ind);
				array_index_image++;
				x_ind = x_ind - 1;
				y_ind = y_ind + 1;
			}
		}
		if (y_ind < matrix_image.rows - 1) {
			array_image.at<float>(0, array_index_image) = matrix_image.at<float>(x_ind, y_ind);
			array_index_image++;
			y_ind = y_ind + 1;

			while (y_ind > 0) {
				array_image.at<float>(0, array_index_image) = matrix_image.at<float>(x_ind, y_ind);
				array_index_image++;
				y_ind = y_ind - 1;
				x_ind = x_ind + 1;
			}
		}
		else if (y_ind == matrix_image.rows - 1) {
			array_image.at<float>(0, array_index_image) = matrix_image.at<float>(x_ind, y_ind);
			array_index_image++;
			x_ind = x_ind + 1;

			while (x_ind < matrix_image.cols - 1) {
				array_image.at<float>(0, array_index_image) = matrix_image.at<float>(x_ind, y_ind);
				array_index_image++;
				y_ind = y_ind - 1;
				x_ind = x_ind + 1;
			}
		}
	}
	return array_image;
}

Mat ZigZagger_invert(Mat array_image) {
	Mat matrix_image = Mat::zeros(sqrt(array_image.cols), sqrt(array_image.cols), CV_32F);

	int array_index_image = 0;

	int x_ind = 0;
	int y_ind = 0;

	while (array_index_image < (matrix_image.cols * matrix_image.rows)) {
		if (x_ind < matrix_image.cols - 1) {
			matrix_image.at<float>(x_ind, y_ind) = array_image.at<float>(0, array_index_image);
			array_index_image++;
			x_ind = x_ind + 1;

			while (x_ind > 0) {
				matrix_image.at<float>(x_ind, y_ind) = array_image.at<float>(0, array_index_image);
				array_index_image++;
				x_ind = x_ind - 1;
				y_ind = y_ind + 1;
			}
		}
		else if (x_ind == matrix_image.cols - 1) {
			matrix_image.at<float>(x_ind, y_ind) = array_image.at<float>(0, array_index_image);
			array_index_image++;
			y_ind = y_ind + 1;

			while (y_ind < matrix_image.rows - 1) {
				matrix_image.at<float>(x_ind, y_ind) = array_image.at<float>(0, array_index_image);
				array_index_image++;
				x_ind = x_ind - 1;
				y_ind = y_ind + 1;
			}
		}
		if (y_ind < matrix_image.rows - 1) {
			matrix_image.at<float>(x_ind, y_ind) = array_image.at<float>(0, array_index_image);
			array_index_image++;
			y_ind = y_ind + 1;

			while (y_ind > 0) {
				matrix_image.at<float>(x_ind, y_ind) = array_image.at<float>(0, array_index_image);
				array_index_image++;
				y_ind = y_ind - 1;
				x_ind = x_ind + 1;
			}
		}
		else if (y_ind == matrix_image.rows - 1) {
			matrix_image.at<float>(x_ind, y_ind) = array_image.at<float>(0, array_index_image);
			array_index_image++;
			x_ind = x_ind + 1;

			while (x_ind < matrix_image.cols - 1) {
				matrix_image.at<float>(x_ind, y_ind) = array_image.at<float>(0, array_index_image);
				array_index_image++;
				y_ind = y_ind - 1;
				x_ind = x_ind + 1;
			}
		}
	}
	return matrix_image;
}

Mat run_length_encoding(Mat array_image) {
	Mat array_image_coded;

	float rep_total = 1;
	float nums_non_con_total = 0;

	for (int i = 0; i < array_image.cols; i++) {
		float curr;
		float next;

		if (i < array_image.cols - 2) {
			curr = array_image.at<float>(0, i);
			next = array_image.at<float>(0, i + 1);
		}
		else {
			curr = array_image.at<float>(0, i);
			next = INT_MAX;
		}

		if (curr == next) {
			rep_total++;
		}
		else {
			nums_non_con_total++;
			rep_total = 1;
		}
	}

	array_image_coded = Mat(2, nums_non_con_total, CV_32F, float(0));
	nums_non_con_total = 0;

	for (int i = 0; i < array_image.cols; i++) {
		float curr;
		float next;

		if (i < array_image.cols - 2) {
			curr = array_image.at<float>(0, i);
			next = array_image.at<float>(0, i + 1);
		}
		else {
			curr = array_image.at<float>(0, i);
			next = INT_MAX;
		}

		if (curr == next) {
			rep_total++;
		}
		else {
			array_image_coded.at<float>(0, nums_non_con_total) = curr;
			array_image_coded.at<float>(1, nums_non_con_total) = rep_total;
			nums_non_con_total++;
			rep_total = 1;
		}
	}

	return array_image_coded;
}

Mat run_length_decoding(Mat array_image_coded) {
	Mat array_image;

	float rep_total = 0;

	for (int i = 0; i < array_image_coded.cols; i++)
		rep_total = rep_total + array_image_coded.at<float>(1, i);

	array_image = Mat(1, rep_total, CV_32F, float(0));

	int array_index_image = 0;

	for (int i = 0; i < array_image_coded.cols; i++) {
		float val = array_image_coded.at<float>(0, i);
		float num_rep = array_image_coded.at<float>(1, i);

		for (int j = 0; j < num_rep; j++) {
			array_image.at<float>(0, array_index_image) = val;
			array_index_image++;
		}
	}

	return array_image;
}

int main()
{
	cudaFree(0);

	Mat imagen;

	bool quantization;

	float* imagen_original, * d_imagen_original, * d_mask;
	float* d_imagen_trans_f1, * d_imagen_trans_f2, * d_imagen_trans_f3, * d_imagen_trans_f4;
	float* imagen_trans_f1_GPU;

	// Enter the path of image to compress HEREPATH
	const string orig_Imagen = "/home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/img/image2.tif";
	

	imagen = imread(orig_Imagen.c_str(), IMREAD_GRAYSCALE);
	imagen.convertTo(imagen, CV_32F);

	if (imagen.empty()) {
		cout << "Image not found.";
		return 0;
	}

	int frows = imagen.rows;
	int fcolumns = imagen.cols;
	const size_t no_of_pixels = frows * fcolumns;

	float mask[LONG_MASK] =
	{
		0.3566, 0.3566, 0.5000, 0, 0.7141, 0, 0, 0,
		0.3566, 0.3566, 0.5000, 0, -0.7141, 0, 0, 0,
		0.3566, 0.3566, -0.5000, 0, 0, 0.7141, 0, 0,
		0.3566, 0.3566, -0.5000, 0, 0, -0.7141, 0, 0,
		0.3566, -0.3566, 0, 0.5000, 0, 0, 0.7141, 0,
		0.3566, -0.3566, 0, 0.5000, 0, 0, -0.7141, 0,
		0.3566, -0.3566, 0, -0.5000, 0, 0, 0, 0.7141,
		0.3566, -0.3566, 0, -0.5000, 0, 0, 0, -0.7141
	};

	float mask_inv[LONG_MASK] =
	{
		0.3566, 0.3566, 0.3566, 0.3566, 0.3566, 0.3566, 0.3566, 0.3566,
		0.3566, 0.3566, 0.3566, 0.3566, -0.3566, -0.3566, -0.3566, -0.3566,
		0.5000, 0.5000, -0.5000, -0.5000, 0, 0, 0, 0,
		0, 0, 0, 0, 0.5000, 0.5000, -0.5000, -0.5000,
		0.7141, -0.7141, 0, 0, 0, 0, 0, 0,
		0, 0, 0.7141, -0.7141, 0, 0, 0, 0,
		0, 0, 0, 0, 0.7141, -0.7141, 0, 0,
		0, 0, 0, 0, 0, 0, 0.7141, -0.7141
	};

	imagen_original = (float*)imagen.ptr<float>(0);
	imagen_trans_f1_GPU = (float*)malloc(sizeof(float) * no_of_pixels);

	int N = DIM_MASK, M = DIM_MASK;

	const dim3 gridSize(frows / M, fcolumns / N, 1);
	const dim3 blockSize(M, N, 1);

	cudaMalloc(&d_imagen_original, sizeof(float) * no_of_pixels);
	cudaMalloc(&d_mask, sizeof(float) * LONG_MASK);
	cudaMalloc(&d_imagen_trans_f1, sizeof(float) * no_of_pixels);
	cudaMalloc(&d_imagen_trans_f2, sizeof(float) * no_of_pixels);
	cudaMalloc(&d_imagen_trans_f3, sizeof(float) * no_of_pixels);
	cudaMalloc(&d_imagen_trans_f4, sizeof(float) * no_of_pixels);

	cudaMemcpy(d_imagen_original, imagen_original, sizeof(float) * no_of_pixels, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask_inv, sizeof(float) * LONG_MASK, cudaMemcpyHostToDevice);

	clock_t timer_1 = clock();

	quantization = true;

	HaarWaveletWrapper::MultDerechaGPU(gridSize, blockSize, d_imagen_original, d_imagen_trans_f1, d_mask,
		frows, fcolumns, quantization);

	cudaMemcpy(d_mask, mask, sizeof(float) * LONG_MASK, cudaMemcpyHostToDevice);

	HaarWaveletWrapper::MultIzquierdaGPU(gridSize, blockSize, d_imagen_trans_f1, d_imagen_trans_f2, d_mask,
		frows, fcolumns, quantization);

	cudaDeviceSynchronize();
	cudaMemcpy(imagen_trans_f1_GPU, d_imagen_trans_f2, sizeof(float) * no_of_pixels, cudaMemcpyDeviceToHost);

	Mat imagenWaveletTransformadaGPU(frows, fcolumns, CV_32F, imagen_trans_f1_GPU);

	Mat array_image = zigZagger(imagenWaveletTransformadaGPU);

	Mat ImageArrayCompressed = run_length_encoding(array_image);

	int InputBitcost = imagen.rows * imagen.cols * 8;
	cout << "InputBitcost: " << InputBitcost << endl;

	float OutputBitcost = ImageArrayCompressed.rows * ImageArrayCompressed.cols * 8;
	cout << "OutputBitcost: " << OutputBitcost << endl;


	array_image = run_length_decoding(ImageArrayCompressed);

	imagenWaveletTransformadaGPU = ZigZagger_invert(array_image);

	cudaMemcpy(d_imagen_original, (float*)imagenWaveletTransformadaGPU.ptr<float>(0), sizeof(float) * no_of_pixels, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask, sizeof(float) * LONG_MASK, cudaMemcpyHostToDevice);

	quantization = false;

	HaarWaveletWrapper::MultDerechaGPU(gridSize, blockSize, d_imagen_original, d_imagen_trans_f1, d_mask,
		frows, fcolumns, quantization);

	cudaMemcpy(d_mask, mask_inv, sizeof(float) * LONG_MASK, cudaMemcpyHostToDevice);

	HaarWaveletWrapper::MultIzquierdaGPU(gridSize, blockSize, d_imagen_trans_f1, d_imagen_trans_f2, d_mask,
		frows, fcolumns, quantization);

	cudaDeviceSynchronize();
	cudaMemcpy(imagen_trans_f1_GPU, d_imagen_trans_f2, sizeof(float) * no_of_pixels, cudaMemcpyDeviceToHost);

	timer_1 = clock() - timer_1;

	printf("Size of Image: [%d, %d]\n", frows, fcolumns);
	//printf("Time taken by GPU %10.3f ms.\n", ((timer_1) / double(CLOCKS_PER_SEC) * 1000));

	Mat imagen_compressed(frows, fcolumns, CV_32F, imagen_trans_f1_GPU);

	imagen_compressed.convertTo(imagen_compressed, CV_64F);
	imagen.convertTo(imagen, CV_64F);

	// cout << "Initial Image Size: " << getMatSizeInKB(imagen) << " KB" << endl;
	// cout << "New Image Size: " << getMatSizeInKB(imagen_compressed) << " KB" << endl;

	cout << "Compression Ratio: " << InputBitcost / OutputBitcost << endl;

	cout << "Mean Square Error: " << qm::mean_square_error(imagen, imagen_compressed) << endl;
	
	cout << "Peak Signal Noise Ratio: " << qm::peak_signal_noise_ratio(imagen, imagen_compressed, 1) << endl;

	cout << "Structural Similarity Index Measure: " << qm::structural_similarity_index_measure(imagen, imagen_compressed, 1) << endl;

	imagen_compressed.convertTo(imagen_compressed, CV_8UC1);
	imagen.convertTo(imagen, CV_8UC1);
	cout<< imagen.elemSize()<<" "<<imagen_compressed.elemSize()<<endl;

	std::ifstream fileB(orig_Imagen, std::ios::binary | std::ios::ate);

    // Check if the file was opened successfully
    if (!fileB) {
        std::cerr << "Unable to open file";
        return 1;  
    }

    // Get the size of the file
    std::streamsize size2 = fileB.tellg();
    fileB.close();

    // Print the size of the file
    std::cout << "Size of the original file is " << size2 << " bytes.\n";

	// Define the path where the compressed image will be saved
	string outputPath = "/home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/img/compressedImage.tif";

	// Save the compressed image to the defined path
	imwrite(outputPath, imagen_compressed);

	std::ifstream file("/home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/img/compressedImage.tif", std::ios::binary | std::ios::ate);

    // Check if the file was opened successfully
    if (!file) {
        std::cerr << "Unable to open file";
        return 1;  
    }

    // Get the size of the file
    std::streamsize size = file.tellg();
    file.close();

    // Print the size of the file
    std::cout << "Size of the compressed file is " << size << " bytes.\n";

	imshow("Original Picture", imagen);
	imshow("Compressed Picture", imagen_compressed);
	waitKey(0);

	return 0;
};