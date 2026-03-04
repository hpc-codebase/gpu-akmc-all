#pragma once

#include <mpi.h>

bool gpu_comm_enabled();
void gpu_comm_init(MPI_Comm comm);
void gpu_comm_finalize();

void gpu_comm_sanity_allreduce(MPI_Comm comm);

