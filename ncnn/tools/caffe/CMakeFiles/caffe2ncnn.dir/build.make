# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/dm/data3/release/STMC/ncnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dm/data3/release/STMC/ncnn/build

# Include any dependencies generated for this target.
include tools/caffe/CMakeFiles/caffe2ncnn.dir/depend.make

# Include the progress variables for this target.
include tools/caffe/CMakeFiles/caffe2ncnn.dir/progress.make

# Include the compile flags for this target's objects.
include tools/caffe/CMakeFiles/caffe2ncnn.dir/flags.make

tools/caffe/caffe.pb.cc: ../tools/caffe/caffe.proto
tools/caffe/caffe.pb.cc: /usr/local/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/dm/data3/release/STMC/ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running C++ protocol buffer compiler on caffe.proto"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/caffe && /usr/local/bin/protoc --cpp_out=/home/dm/data3/release/STMC/ncnn/build/tools/caffe -I /home/dm/data3/release/STMC/ncnn/tools/caffe /home/dm/data3/release/STMC/ncnn/tools/caffe/caffe.proto

tools/caffe/caffe.pb.h: tools/caffe/caffe.pb.cc
	@$(CMAKE_COMMAND) -E touch_nocreate tools/caffe/caffe.pb.h

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o: tools/caffe/CMakeFiles/caffe2ncnn.dir/flags.make
tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o: ../tools/caffe/caffe2ncnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dm/data3/release/STMC/ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/caffe && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o -c /home/dm/data3/release/STMC/ncnn/tools/caffe/caffe2ncnn.cpp

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.i"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dm/data3/release/STMC/ncnn/tools/caffe/caffe2ncnn.cpp > CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.i

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.s"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dm/data3/release/STMC/ncnn/tools/caffe/caffe2ncnn.cpp -o CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.s

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.requires:

.PHONY : tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.requires

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.provides: tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.requires
	$(MAKE) -f tools/caffe/CMakeFiles/caffe2ncnn.dir/build.make tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.provides.build
.PHONY : tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.provides

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.provides.build: tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o


tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o: tools/caffe/CMakeFiles/caffe2ncnn.dir/flags.make
tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o: tools/caffe/caffe.pb.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dm/data3/release/STMC/ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/caffe && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o -c /home/dm/data3/release/STMC/ncnn/build/tools/caffe/caffe.pb.cc

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.i"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dm/data3/release/STMC/ncnn/build/tools/caffe/caffe.pb.cc > CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.i

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.s"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/caffe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dm/data3/release/STMC/ncnn/build/tools/caffe/caffe.pb.cc -o CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.s

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.requires:

.PHONY : tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.requires

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.provides: tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.requires
	$(MAKE) -f tools/caffe/CMakeFiles/caffe2ncnn.dir/build.make tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.provides.build
.PHONY : tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.provides

tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.provides.build: tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o


# Object files for target caffe2ncnn
caffe2ncnn_OBJECTS = \
"CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o" \
"CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o"

# External object files for target caffe2ncnn
caffe2ncnn_EXTERNAL_OBJECTS =

tools/caffe/caffe2ncnn: tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o
tools/caffe/caffe2ncnn: tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o
tools/caffe/caffe2ncnn: tools/caffe/CMakeFiles/caffe2ncnn.dir/build.make
tools/caffe/caffe2ncnn: /usr/local/lib/libprotobuf.so
tools/caffe/caffe2ncnn: tools/caffe/CMakeFiles/caffe2ncnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dm/data3/release/STMC/ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable caffe2ncnn"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/caffe && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/caffe2ncnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/caffe/CMakeFiles/caffe2ncnn.dir/build: tools/caffe/caffe2ncnn

.PHONY : tools/caffe/CMakeFiles/caffe2ncnn.dir/build

tools/caffe/CMakeFiles/caffe2ncnn.dir/requires: tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o.requires
tools/caffe/CMakeFiles/caffe2ncnn.dir/requires: tools/caffe/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o.requires

.PHONY : tools/caffe/CMakeFiles/caffe2ncnn.dir/requires

tools/caffe/CMakeFiles/caffe2ncnn.dir/clean:
	cd /home/dm/data3/release/STMC/ncnn/build/tools/caffe && $(CMAKE_COMMAND) -P CMakeFiles/caffe2ncnn.dir/cmake_clean.cmake
.PHONY : tools/caffe/CMakeFiles/caffe2ncnn.dir/clean

tools/caffe/CMakeFiles/caffe2ncnn.dir/depend: tools/caffe/caffe.pb.cc
tools/caffe/CMakeFiles/caffe2ncnn.dir/depend: tools/caffe/caffe.pb.h
	cd /home/dm/data3/release/STMC/ncnn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dm/data3/release/STMC/ncnn /home/dm/data3/release/STMC/ncnn/tools/caffe /home/dm/data3/release/STMC/ncnn/build /home/dm/data3/release/STMC/ncnn/build/tools/caffe /home/dm/data3/release/STMC/ncnn/build/tools/caffe/CMakeFiles/caffe2ncnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/caffe/CMakeFiles/caffe2ncnn.dir/depend

