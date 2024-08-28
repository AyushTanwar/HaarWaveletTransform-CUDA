# X-Ray Image  Compression using Haar-Wavelet and CUDA

I created a Haar Wavelet based Image Compression Model, that uses CUDA to expedite the processing time of image compression. The Images use 80% less storage compared to traditional non compressed images. The model is optimized for X-Ray Images, but by tweaking the paramaeters and masks, it can be used for other type of images too.


My Reference: [Haar Wavelet Image Compression And Quality Evaluation](https://arxiv.org/pdf/1010.4084.pdf)

## How to Use

In ```HWIC.cpp``` file Replace the following file path with the path of your own image
```
    const string orig_Imagen = "/home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/img/IM-0001-0007.tif";
```
For easier search for the line with path use the search term ```HEREPATH``` 

## Concurrent path
The the concurrent matrix computation CUDA code is in ```src``` folder
