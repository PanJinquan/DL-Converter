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
include tools/mxnet/CMakeFiles/mxnet2ncnn.dir/depend.make

# Include the progress variables for this target.
include tools/mxnet/CMakeFiles/mxnet2ncnn.dir/progress.make

# Include the compile flags for this target's objects.
include tools/mxnet/CMakeFiles/mxnet2ncnn.dir/flags.make

tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o: tools/mxnet/CMakeFiles/mxnet2ncnn.dir/flags.make
tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o: ../tools/mxnet/mxnet2ncnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dm/data3/release/STMC/ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/mxnet && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o -c /home/dm/data3/release/STMC/ncnn/tools/mxnet/mxnet2ncnn.cpp

tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.i"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/mxnet && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dm/data3/release/STMC/ncnn/tools/mxnet/mxnet2ncnn.cpp > CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.i

tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.s"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/mxnet && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dm/data3/release/STMC/ncnn/tools/mxnet/mxnet2ncnn.cpp -o CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.s

tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.requires:

.PHONY : tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.requires

tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.provides: tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.requires
	$(MAKE) -f tools/mxnet/CMakeFiles/mxnet2ncnn.dir/build.make tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.provides.build
.PHONY : tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.provides

tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.provides.build: tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o


# Object files for target mxnet2ncnn
mxnet2ncnn_OBJECTS = \
"CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o"

# External object files for target mxnet2ncnn
mxnet2ncnn_EXTERNAL_OBJECTS =

tools/mxnet/mxnet2ncnn: tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o
tools/mxnet/mxnet2ncnn: tools/mxnet/CMakeFiles/mxnet2ncnn.dir/build.make
tools/mxnet/mxnet2ncnn: tools/mxnet/CMakeFiles/mxnet2ncnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dm/data3/release/STMC/ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mxnet2ncnn"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/mxnet && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mxnet2ncnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/mxnet/CMakeFiles/mxnet2ncnn.dir/build: tools/mxnet/mxnet2ncnn

.PHONY : tools/mxnet/CMakeFiles/mxnet2ncnn.dir/build

tools/mxnet/CMakeFiles/mxnet2ncnn.dir/requires: tools/mxnet/CMakeFiles/mxnet2ncnn.dir/mxnet2ncnn.cpp.o.requires

.PHONY : tools/mxnet/CMakeFiles/mxnet2ncnn.dir/requires

tools/mxnet/CMakeFiles/mxnet2ncnn.dir/clean:
	cd /home/dm/data3/release/STMC/ncnn/build/tools/mxnet && $(CMAKE_COMMAND) -P CMakeFiles/mxnet2ncnn.dir/cmake_clean.cmake
.PHONY : tools/mxnet/CMakeFiles/mxnet2ncnn.dir/clean

tools/mxnet/CMakeFiles/mxnet2ncnn.dir/depend:
	cd /home/dm/data3/release/STMC/ncnn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dm/data3/release/STMC/ncnn /home/dm/data3/release/STMC/ncnn/tools/mxnet /home/dm/data3/release/STMC/ncnn/build /home/dm/data3/release/STMC/ncnn/build/tools/mxnet /home/dm/data3/release/STMC/ncnn/build/tools/mxnet/CMakeFiles/mxnet2ncnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/mxnet/CMakeFiles/mxnet2ncnn.dir/depend

