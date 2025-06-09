set(TRTLLM_NANOBIND_MODULE nano_bindings)
set(TRTLLM_NANOBIND_MODULE
    ${TRTLLM_NANOBIND_MODULE}
    PARENT_SCOPE)

set(SRCS executor/request.cpp executor/bindings.cpp bindings.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

nanobind_add_module(${TRTLLM_NANOBIND_MODULE} ${SRCS})
message(STATUS "nanobind_add_module ${TRTLLM_NANOBIND_MODULE} ${SRCS}")

set_property(TARGET ${TRTLLM_NANOBIND_MODULE} PROPERTY POSITION_INDEPENDENT_CODE
                                                       ON)

target_link_directories(${TRTLLM_NANOBIND_MODULE} PUBLIC
                        "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(
  ${TRTLLM_NANOBIND_MODULE}
  PUBLIC ${SHARED_TARGET} ${UNDEFINED_FLAG} ${NO_AS_NEEDED_FLAG}
         ${Python3_LIBRARIES} ${TORCH_LIBRARIES} torch_python)
target_compile_definitions(
  ${TRTLLM_NANOBIND_MODULE}
  PUBLIC TRTLLM_NANOBIND_MODULE=${TRTLLM_NANOBIND_MODULE})

if(NOT WIN32)
  set_target_properties(
    ${TRTLLM_NANOBIND_MODULE}
    PROPERTIES
      LINK_FLAGS
      "-Wl,-rpath,'$ORIGIN/libs' -Wl,-rpath,'$ORIGIN/../nvidia/nccl/lib' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}"
  )
endif()
