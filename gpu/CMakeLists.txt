# set binary path and lib storage path for compiled binary.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_library(${KMC_GPU_LIB_NAME} STATIC
        gpu_simulate.hip
        gpu_simulate.h
        gpuError.hip
        gpuError.h
        DeviceVacRatesSolver.h
        DeviceVacRatesSolver.cpp
        gpu_comm.cpp
        gpu_comm.h
        rccl_smoke_test.cpp
        rccl_smoke_test.h
        )

target_compile_options(${KMC_GPU_LIB_NAME} PRIVATE -D__HIP_PLATFORM_HCC__= -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP  -D__HIP__ -D__HIP_PLATFORM_AMD__= -I"${ROCM_PATH}/hip/include" -I"${ROCM_PATH}/llvm/lib/clang/17.0.0" -I${ROCM_PATH}/hsa/include)

target_include_directories(${KMC_GPU_LIB_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        )

if(KMC_RCCL_ENABLE_FLAG)
        find_path(KMC_NCCL_INCLUDE_DIR NAMES nccl.h rccl.h
                PATHS
                ENV RCCL_ROOT
                ENV ROCM_PATH
                ${ROCM_PATH}
                /opt/rocm
                /opt
                PATH_SUFFIXES include)
        find_library(KMC_NCCL_LIBRARY NAMES rccl nccl
                PATHS
                ENV RCCL_ROOT
                ENV ROCM_PATH
                ${ROCM_PATH}
                /opt/rocm
                /opt
                PATH_SUFFIXES lib lib64)
        if(NOT KMC_NCCL_INCLUDE_DIR OR NOT KMC_NCCL_LIBRARY)
                message(FATAL_ERROR "KMC_RCCL_ENABLE_FLAG=ON but NCCL/RCCL headers or libraries not found. Set RCCL_ROOT or ROCM_PATH.")
        endif()
        target_include_directories(${KMC_GPU_LIB_NAME} PRIVATE ${KMC_NCCL_INCLUDE_DIR})
        target_link_libraries(${KMC_GPU_LIB_NAME} PUBLIC ${KMC_NCCL_LIBRARY})
        target_compile_definitions(${KMC_GPU_LIB_NAME} PRIVATE KMC_RCCL_ENABLE=1)
endif()

target_link_libraries(${KMC_GPU_LIB_NAME} PUBLIC ${PKMC_LIB_NAME})

# install bin and libs
install(TARGETS ${KMC_GPU_LIB_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

# install header.
install(DIRECTORY "./"
        DESTINATION "include"
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        )
