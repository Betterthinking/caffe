# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yuxi/ker2col-caffe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yuxi/ker2col-caffe/build

# Include any dependencies generated for this target.
include tools/CMakeFiles/ssd_detect.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/ssd_detect.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/ssd_detect.dir/flags.make

tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o: tools/CMakeFiles/ssd_detect.dir/flags.make
tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o: ../tools/ssd_detect.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yuxi/ker2col-caffe/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o"
	cd /home/yuxi/ker2col-caffe/build/tools && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o -c /home/yuxi/ker2col-caffe/tools/ssd_detect.cpp

tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ssd_detect.dir/ssd_detect.cpp.i"
	cd /home/yuxi/ker2col-caffe/build/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yuxi/ker2col-caffe/tools/ssd_detect.cpp > CMakeFiles/ssd_detect.dir/ssd_detect.cpp.i

tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ssd_detect.dir/ssd_detect.cpp.s"
	cd /home/yuxi/ker2col-caffe/build/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yuxi/ker2col-caffe/tools/ssd_detect.cpp -o CMakeFiles/ssd_detect.dir/ssd_detect.cpp.s

tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o.requires:

.PHONY : tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o.requires

tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o.provides: tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/ssd_detect.dir/build.make tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o.provides.build
.PHONY : tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o.provides

tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o.provides.build: tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o


# Object files for target ssd_detect
ssd_detect_OBJECTS = \
"CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o"

# External object files for target ssd_detect
ssd_detect_EXTERNAL_OBJECTS =

tools/ssd_detect: tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o
tools/ssd_detect: tools/CMakeFiles/ssd_detect.dir/build.make
tools/ssd_detect: lib/libcaffe.so.1.0.0-rc3
tools/ssd_detect: lib/libproto.a
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_system.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_regex.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libglog.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libsz.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libz.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libdl.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libm.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libglog.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libsz.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libz.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libdl.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libm.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/liblmdb.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libleveldb.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libsnappy.so
tools/ssd_detect: /usr/local/lib/libopencv_highgui.so.3.3.0
tools/ssd_detect: /usr/local/lib/libopencv_videoio.so.3.3.0
tools/ssd_detect: /usr/local/lib/libopencv_imgcodecs.so.3.3.0
tools/ssd_detect: /usr/local/lib/libopencv_imgproc.so.3.3.0
tools/ssd_detect: /usr/local/lib/libopencv_core.so.3.3.0
tools/ssd_detect: /usr/lib/libopenblas.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libpython2.7.so
tools/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_python.so
tools/ssd_detect: tools/CMakeFiles/ssd_detect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yuxi/ker2col-caffe/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ssd_detect"
	cd /home/yuxi/ker2col-caffe/build/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ssd_detect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/ssd_detect.dir/build: tools/ssd_detect

.PHONY : tools/CMakeFiles/ssd_detect.dir/build

tools/CMakeFiles/ssd_detect.dir/requires: tools/CMakeFiles/ssd_detect.dir/ssd_detect.cpp.o.requires

.PHONY : tools/CMakeFiles/ssd_detect.dir/requires

tools/CMakeFiles/ssd_detect.dir/clean:
	cd /home/yuxi/ker2col-caffe/build/tools && $(CMAKE_COMMAND) -P CMakeFiles/ssd_detect.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/ssd_detect.dir/clean

tools/CMakeFiles/ssd_detect.dir/depend:
	cd /home/yuxi/ker2col-caffe/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yuxi/ker2col-caffe /home/yuxi/ker2col-caffe/tools /home/yuxi/ker2col-caffe/build /home/yuxi/ker2col-caffe/build/tools /home/yuxi/ker2col-caffe/build/tools/CMakeFiles/ssd_detect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/ssd_detect.dir/depend

