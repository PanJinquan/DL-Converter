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
include tools/quantize/CMakeFiles/ncnn2table.dir/depend.make

# Include the progress variables for this target.
include tools/quantize/CMakeFiles/ncnn2table.dir/progress.make

# Include the compile flags for this target's objects.
include tools/quantize/CMakeFiles/ncnn2table.dir/flags.make

tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o: tools/quantize/CMakeFiles/ncnn2table.dir/flags.make
tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o: ../tools/quantize/ncnn2table.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dm/data3/release/STMC/ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/quantize && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o -c /home/dm/data3/release/STMC/ncnn/tools/quantize/ncnn2table.cpp

tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ncnn2table.dir/ncnn2table.cpp.i"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/quantize && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dm/data3/release/STMC/ncnn/tools/quantize/ncnn2table.cpp > CMakeFiles/ncnn2table.dir/ncnn2table.cpp.i

tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ncnn2table.dir/ncnn2table.cpp.s"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/quantize && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dm/data3/release/STMC/ncnn/tools/quantize/ncnn2table.cpp -o CMakeFiles/ncnn2table.dir/ncnn2table.cpp.s

tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o.requires:

.PHONY : tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o.requires

tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o.provides: tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o.requires
	$(MAKE) -f tools/quantize/CMakeFiles/ncnn2table.dir/build.make tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o.provides.build
.PHONY : tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o.provides

tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o.provides.build: tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o


# Object files for target ncnn2table
ncnn2table_OBJECTS = \
"CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o"

# External object files for target ncnn2table
ncnn2table_EXTERNAL_OBJECTS =

tools/quantize/ncnn2table: tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o
tools/quantize/ncnn2table: tools/quantize/CMakeFiles/ncnn2table.dir/build.make
tools/quantize/ncnn2table: src/libncnn.a
tools/quantize/ncnn2table: /usr/local/lib/libopencv_highgui.so.4.3.0
tools/quantize/ncnn2table: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
tools/quantize/ncnn2table: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/quantize/ncnn2table: /usr/local/lib/libopencv_videoio.so.4.3.0
tools/quantize/ncnn2table: /usr/local/lib/libopencv_imgcodecs.so.4.3.0
tools/quantize/ncnn2table: /usr/local/lib/libopencv_imgproc.so.4.3.0
tools/quantize/ncnn2table: /usr/local/lib/libopencv_core.so.4.3.0
tools/quantize/ncnn2table: tools/quantize/CMakeFiles/ncnn2table.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dm/data3/release/STMC/ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ncnn2table"
	cd /home/dm/data3/release/STMC/ncnn/build/tools/quantize && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ncnn2table.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/quantize/CMakeFiles/ncnn2table.dir/build: tools/quantize/ncnn2table

.PHONY : tools/quantize/CMakeFiles/ncnn2table.dir/build

tools/quantize/CMakeFiles/ncnn2table.dir/requires: tools/quantize/CMakeFiles/ncnn2table.dir/ncnn2table.cpp.o.requires

.PHONY : tools/quantize/CMakeFiles/ncnn2table.dir/requires

tools/quantize/CMakeFiles/ncnn2table.dir/clean:
	cd /home/dm/data3/release/STMC/ncnn/build/tools/quantize && $(CMAKE_COMMAND) -P CMakeFiles/ncnn2table.dir/cmake_clean.cmake
.PHONY : tools/quantize/CMakeFiles/ncnn2table.dir/clean

tools/quantize/CMakeFiles/ncnn2table.dir/depend:
	cd /home/dm/data3/release/STMC/ncnn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dm/data3/release/STMC/ncnn /home/dm/data3/release/STMC/ncnn/tools/quantize /home/dm/data3/release/STMC/ncnn/build /home/dm/data3/release/STMC/ncnn/build/tools/quantize /home/dm/data3/release/STMC/ncnn/build/tools/quantize/CMakeFiles/ncnn2table.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/quantize/CMakeFiles/ncnn2table.dir/depend
