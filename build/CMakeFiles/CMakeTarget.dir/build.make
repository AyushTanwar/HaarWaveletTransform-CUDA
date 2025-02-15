# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/build

# Include any dependencies generated for this target.
include CMakeFiles/CMakeTarget.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/CMakeTarget.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/CMakeTarget.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CMakeTarget.dir/flags.make

CMakeFiles/CMakeTarget.dir/HWIC.cpp.o: CMakeFiles/CMakeTarget.dir/flags.make
CMakeFiles/CMakeTarget.dir/HWIC.cpp.o: ../HWIC.cpp
CMakeFiles/CMakeTarget.dir/HWIC.cpp.o: CMakeFiles/CMakeTarget.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CMakeTarget.dir/HWIC.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CMakeTarget.dir/HWIC.cpp.o -MF CMakeFiles/CMakeTarget.dir/HWIC.cpp.o.d -o CMakeFiles/CMakeTarget.dir/HWIC.cpp.o -c /home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/HWIC.cpp

CMakeFiles/CMakeTarget.dir/HWIC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CMakeTarget.dir/HWIC.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/HWIC.cpp > CMakeFiles/CMakeTarget.dir/HWIC.cpp.i

CMakeFiles/CMakeTarget.dir/HWIC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CMakeTarget.dir/HWIC.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/HWIC.cpp -o CMakeFiles/CMakeTarget.dir/HWIC.cpp.s

CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.o: CMakeFiles/CMakeTarget.dir/flags.make
CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.o: ../src/HWCuda.cu
CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.o: CMakeFiles/CMakeTarget.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.o -MF CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.o.d -x cu -c /home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/src/HWCuda.cu -o CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.o

CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target CMakeTarget
CMakeTarget_OBJECTS = \
"CMakeFiles/CMakeTarget.dir/HWIC.cpp.o" \
"CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.o"

# External object files for target CMakeTarget
CMakeTarget_EXTERNAL_OBJECTS =

CMakeTarget: CMakeFiles/CMakeTarget.dir/HWIC.cpp.o
CMakeTarget: CMakeFiles/CMakeTarget.dir/src/HWCuda.cu.o
CMakeTarget: CMakeFiles/CMakeTarget.dir/build.make
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
CMakeTarget: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
CMakeTarget: CMakeFiles/CMakeTarget.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable CMakeTarget"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CMakeTarget.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CMakeTarget.dir/build: CMakeTarget
.PHONY : CMakeFiles/CMakeTarget.dir/build

CMakeFiles/CMakeTarget.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CMakeTarget.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CMakeTarget.dir/clean

CMakeFiles/CMakeTarget.dir/depend:
	cd /home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression /home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression /home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/build /home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/build /home/vipul/Downloads/ayush/Haar-Wavelet-Image-Compression/build/CMakeFiles/CMakeTarget.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CMakeTarget.dir/depend

