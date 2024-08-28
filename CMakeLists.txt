# CMakeList.txt : CMake project for HWIC, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project ("HWIC" LANGUAGES CXX CUDA)

FIND_PACKAGE( OpenCV REQUIRED )                              
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS})

# Add source to this project's executable.
add_executable (CMakeTarget "HWIC.cpp" "src/HWCuda.cu")
TARGET_LINK_LIBRARIES (CMakeTarget ${OpenCV_LIBS})    
TARGET_LINK_LIBRARIES (CMakeTarget ${CUDA_LIBS})    
