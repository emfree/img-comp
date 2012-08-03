project( random_order )
find_package( OpenCV REQUIRED )
add_executable( random_order random_order.cpp )
target_link_libraries( random_order ${OpenCV_LIBS} )

set(CMAKE_BUILD_TYPE Debug)
ADD_DEFINITIONS(-pg)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")

