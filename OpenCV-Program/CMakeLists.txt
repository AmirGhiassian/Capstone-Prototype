cmake_minimum_required(VERSION 3.12)
project(VistaCart)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Link OpenCV to your target

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add your source files
add_executable(VistaCart src/main.cpp)

# Add any additional libraries or dependencies
target_link_libraries(VistaCart ${OpenCV_LIBS})