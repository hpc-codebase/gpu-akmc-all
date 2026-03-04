module use /public/software/sghpc_sdk/modulefiles/
module load sghpc-mpi-gcc/26.3
export CMAKE_PREFIX_PATH=${ROCM_PATH}/dcc/comgr/lib64/cmake/amd_comgr/:$CMAKE_PREFIX_PATH