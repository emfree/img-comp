# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
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

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /auto/sop-nas2a/u/sop-nas2a/vol/home_stars/efreeman/img-comp/img-comp/hierarchy

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /auto/sop-nas2a/u/sop-nas2a/vol/home_stars/efreeman/img-comp/img-comp/hierarchy

# Include any dependencies generated for this target.
include CMakeFiles/random_order.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/random_order.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/random_order.dir/flags.make

CMakeFiles/random_order.dir/random_order.o: CMakeFiles/random_order.dir/flags.make
CMakeFiles/random_order.dir/random_order.o: random_order.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /auto/sop-nas2a/u/sop-nas2a/vol/home_stars/efreeman/img-comp/img-comp/hierarchy/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/random_order.dir/random_order.o"
	/usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/random_order.dir/random_order.o -c /auto/sop-nas2a/u/sop-nas2a/vol/home_stars/efreeman/img-comp/img-comp/hierarchy/random_order.cpp

CMakeFiles/random_order.dir/random_order.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/random_order.dir/random_order.i"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /auto/sop-nas2a/u/sop-nas2a/vol/home_stars/efreeman/img-comp/img-comp/hierarchy/random_order.cpp > CMakeFiles/random_order.dir/random_order.i

CMakeFiles/random_order.dir/random_order.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/random_order.dir/random_order.s"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /auto/sop-nas2a/u/sop-nas2a/vol/home_stars/efreeman/img-comp/img-comp/hierarchy/random_order.cpp -o CMakeFiles/random_order.dir/random_order.s

CMakeFiles/random_order.dir/random_order.o.requires:
.PHONY : CMakeFiles/random_order.dir/random_order.o.requires

CMakeFiles/random_order.dir/random_order.o.provides: CMakeFiles/random_order.dir/random_order.o.requires
	$(MAKE) -f CMakeFiles/random_order.dir/build.make CMakeFiles/random_order.dir/random_order.o.provides.build
.PHONY : CMakeFiles/random_order.dir/random_order.o.provides

CMakeFiles/random_order.dir/random_order.o.provides.build: CMakeFiles/random_order.dir/random_order.o

# Object files for target random_order
random_order_OBJECTS = \
"CMakeFiles/random_order.dir/random_order.o"

# External object files for target random_order
random_order_EXTERNAL_OBJECTS =

random_order: CMakeFiles/random_order.dir/random_order.o
random_order: /usr/local/lib/libopencv_calib3d.so
random_order: /usr/local/lib/libopencv_contrib.so
random_order: /usr/local/lib/libopencv_core.so
random_order: /usr/local/lib/libopencv_features2d.so
random_order: /usr/local/lib/libopencv_flann.so
random_order: /usr/local/lib/libopencv_gpu.so
random_order: /usr/local/lib/libopencv_highgui.so
random_order: /usr/local/lib/libopencv_imgproc.so
random_order: /usr/local/lib/libopencv_legacy.so
random_order: /usr/local/lib/libopencv_ml.so
random_order: /usr/local/lib/libopencv_nonfree.so
random_order: /usr/local/lib/libopencv_objdetect.so
random_order: /usr/local/lib/libopencv_photo.so
random_order: /usr/local/lib/libopencv_stitching.so
random_order: /usr/local/lib/libopencv_ts.so
random_order: /usr/local/lib/libopencv_video.so
random_order: /usr/local/lib/libopencv_videostab.so
random_order: CMakeFiles/random_order.dir/build.make
random_order: CMakeFiles/random_order.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable random_order"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/random_order.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/random_order.dir/build: random_order
.PHONY : CMakeFiles/random_order.dir/build

CMakeFiles/random_order.dir/requires: CMakeFiles/random_order.dir/random_order.o.requires
.PHONY : CMakeFiles/random_order.dir/requires

CMakeFiles/random_order.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/random_order.dir/cmake_clean.cmake
.PHONY : CMakeFiles/random_order.dir/clean

CMakeFiles/random_order.dir/depend:
	cd /auto/sop-nas2a/u/sop-nas2a/vol/home_stars/efreeman/img-comp/img-comp/hierarchy && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /auto/sop-nas2a/u/sop-nas2a/vol/home_stars/efreeman/img-comp/img-comp/hierarchy /auto/sop-nas2a/u/sop-nas2a/vol/home_stars/efreeman/img-comp/img-comp/hierarchy /auto/sop-nas2a/u/sop-nas2a/vol/home_stars/efreeman/img-comp/img-comp/hierarchy /auto/sop-nas2a/u/sop-nas2a/vol/home_stars/efreeman/img-comp/img-comp/hierarchy /auto/sop-nas2a/u/sop-nas2a/vol/home_stars/efreeman/img-comp/img-comp/hierarchy/CMakeFiles/random_order.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/random_order.dir/depend

