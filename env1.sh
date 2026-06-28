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
# module load sghpc-mpi-gcc/26.3

export HIP_PATH=/public/software/sghpc_sdk/Linux_x86_64/26.3/dtk/dtk-25.04.4/hip
export ROCM_PATH=/public/software/sghpc_sdk/Linux_x86_64/26.3/dtk/dtk-25.04.4/


export KMC_RCCL_SMOKE=1
export KMC_GPU_COMM=1
export KMC_RECB_PROFILE=1
# export HSA_FORCE_FINE_GRAIN_PCIE=1


#find /opt/dtk-25.04.1 -iname "*comgr*" -name "*.cmake"
export amd_comgr_DIR=/public/software/sghpc_sdk/Linux_x86_64/26.3/dtk/dtk-25.04.4/lib64/cmake/amd_comgr
export CMAKE_PREFIX_PATH=/public/software/sghpc_sdk/Linux_x86_64/26.3/dtk/dtk-25.04.4:/public/software/sghpc_sdk/Linux_x86_64/26.3/dtk/dtk-25.04.4/lib64/cmake/amd_comgr:$CMAKE_PREFIX_PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/public/software/sghpc_sdk/Linux_x86_64/26.3/dtk/dtk-25.04.4/lib64
export LIBRARY_PATH=/public/home/scnethpc2621/aceaitx8ed/duhao_bw100/misa_kmc_all_gpu/misa-akmc-more/vendor/pkg/github.com/jbeder/yaml-cpp/lib:$LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/public/home/scnethpc2621/aceaitx8ed/duhao_bw100/misa_kmc_all_gpu/misa-akmc-more/vendor/pkg/github.com/jbeder/yaml-cpp/include:$CPLUS_INCLUDE_PATH
export LDFLAGS="-L/public/software/sghpc_sdk/Linux_x86_64/26.3/compilers/gcc-12.4.0/lib64 -Wl,-rpath,/public/software/sghpc_sdk/Linux_x86_64/26.3/compilers/gcc-12.4.0/lib64 -L/public/home/scnethpc2621/aceaitx8ed/duhao_bw100/misa_kmc_all_gpu/misa-akmc-more/vendor/pkg/github.com/jbeder/yaml-cpp/lib"