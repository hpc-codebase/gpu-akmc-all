#########################################################################
# File Name: env.sh
# Author: panzhjie
#########################################################################
#!/bin/bash

module unload mpi/hpcx/2.11.0/gcc-7.3.1
#module unload compiler/devtoolset/7.3.1
#module load compiler/devtoolset/11.2.1
#module load mpi/openmpi/4.0.4/gcc-7.3.1
#module load mpi/hpcx/2.12.0/gcc-7.3.1
#module load compiler/intel/2021.3.0
#module load mpi/intelmpi/2021.3.0
module load compiler/intel/2019.4.243
module load mpi/intelmpi/2021.3.0
module unload compiler/rocm/dtk/22.10.1
#module load compiler/rocm/dtk/21.04
module load compiler/rocm/dtk/24.04.3
#module load mathlib/magma/dtk_23.04
module load compiler/cmake/3.24.1
#module load compiler/cmake/3.16.2
#module load mathlib/magma/dtk_22.10_develop/22.10
#module load mathlib/openblas/0.3.7
#module load mathlib/lapack/intel/3.8.0
