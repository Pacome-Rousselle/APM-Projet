cmake_minimum_required(VERSION 3.17)
project(WaveFunctionCollapse VERSION 0.1 LANGUAGES C CUDA CXX)

find_package(CUDA)
#enlever 
#cmake_policy(SET CMP0009 NEW) # Do not follow symlinks with GLOB_RECURSE

STRING(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" insource)
GET_FILENAME_COMPONENT(PARENTDIR ${CMAKE_SOURCE_DIR} PATH)
STRING(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${PARENTDIR}" insourcesubdir)
if(insource OR insourcesubdir)
    MESSAGE(FATAL_ERROR "${CMAKE_PROJECT_NAME} requires an out of source build")
endif()



# restrict will not work with cpp 
# so define restrict space like wooow 
# 
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
else()
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Pass -fPIC
set(CMAKE_COLOR_MAKEFILE            ON)
set(CMAKE_COLOR_DIAGNOSTICS         ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS   ON) # Always to that...
set(THREADS_PREFER_PTHREAD_FLAG     ON) # Pthread ftw

set(CUDA_SEPARABLE_COMPILATION      ON) #baah c'est important 
#set seperable compilation on 


add_definitions(-DWFC_CUDA)
file(GLOB_RECURSE WaveFunctionCollapse_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu") 
#put everything in .cu 

file(GLOB_RECURSE WaveFunctionCollapse_INC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h")

add_definitions(-DWFC_CUDA)

cuda_add_executable(wfc ${WaveFunctionCollapse_SRC} ${WaveFunctionCollapse_INC} )

#add_executable(wfc ${WaveFunctionCollapse_SRC} ${WaveFunctionCollapse_INC})
target_include_directories(wfc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
target_link_libraries(wfc -pthread -fopenmp -lm)
target_compile_options(wfc PRIVATE -g -O2 -pthread -fopenmp -std=c99) #17 enlever 

#there might be some compilation options that may give error from here 
target_compile_options(wfc PRIVATE
    #-Wall 
    -pedantic
    -Wshadow
    -Wcast-align -Wconversion -Wsign-conversion -Wdouble-promotion
    #-Wunused-variable
    -Wmisleading-indentation
    -Wnull-dereference
    -Wformat=2
)

#virer said 
if(${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
    target_compile_options(wfc PRIVATE
        #-Weverything
        -Wno-vla
        -Wno-unsafe-buffer-usage
        -Wno-declaration-after-statement
    )
elseif(${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(wfc PRIVATE
        -Wno-switch
        
    )
endif()
