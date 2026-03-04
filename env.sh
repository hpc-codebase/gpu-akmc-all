#########################################################################
# File Name: env.sh
# Author: panzhjie
#########################################################################
#!/bin/bash

# module use /public/software/modules/


# module load compiler/intel/2021.3.0 
# module load compiler/cmake/3.25.3 
# module load sghpc-mpi-gcc/25.8
# export CMAKE_PREFIX_PATH=/public/home/sghpc_sdk/Linux_x86_64/25.8/dtk/dtk-25.04.2/dcc/comgr/lib64/cmake/amd_comgr/:$CMAKE_PREFIX_PATH
# module load compiler/intel/2019.4.243
# module load compiler/cmake/3.24.1

# module load mpi/intelmpi/2021.3.0

# module load compiler/rocm/dtk/24.04.3

module load compiler/dtk/25.04.1
module load mpi/hpcx/2.18.0/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1

export HIP_PATH=/opt/dtk-25.04.1
export ROCM_PATH=/opt/dtk-25.04.1


#find /opt/dtk-25.04.1 -iname "*comgr*" -name "*.cmake"
export amd_comgr_DIR=/opt/dtk-25.04.1/lib64/cmake/amd_comgr
export CMAKE_PREFIX_PATH=/opt/dtk-25.04.1:/opt/dtk-25.04.1/lib64/cmake/amd_comgr:$CMAKE_PREFIX_PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/dtk-25.04.1/lib64