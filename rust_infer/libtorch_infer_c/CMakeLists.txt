cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_stream_infer_ffi)

if (NOT CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH "/data/libs/libtorch2.3")
endif()

get_filename_component(libtorch_lib_dir "${CMAKE_PREFIX_PATH}/lib" ABSOLUTE)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(example-app main.cc)
add_library(torch_stream_infer_ffi STATIC libtorch_stream_infer_ffi.cc libtorch_stream_infer_ffi.h)

message("torch_libraries=> ${TORCH_LIBRARIES}")
target_link_libraries(torch_stream_infer_ffi PUBLIC "${TORCH_LIBRARIES}")
set_property(TARGET torch_stream_infer_ffi PROPERTY CXX_STANDARD 17)
target_compile_options(torch_stream_infer_ffi PRIVATE -fPIC)

# useless
set_property(TARGET torch_stream_infer_ffi PROPERTY
  LINK_FLAGS "-Wl,-rpath,${libtorch_lib_dir}")

# CMAKE_PREFIX_PATH
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET example-app
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:example-app>)
# endif (MSVC)