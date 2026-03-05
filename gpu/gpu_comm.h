#pragma once

#include <mpi.h>
#include <cstddef>

bool gpu_comm_enabled();
void gpu_comm_init(MPI_Comm comm);
void gpu_comm_finalize();

void gpu_comm_sanity_allreduce(MPI_Comm comm);

struct GpuCommBuffer {
    void* d_send = nullptr;
    void* d_recv = nullptr;
    size_t send_cap = 0;
    size_t recv_cap = 0;
};

void gpu_comm_buffer_free(GpuCommBuffer& buf);

void gpu_comm_send_recv(GpuCommBuffer& buf,
                        const void* send_host, size_t send_count,
                        void* recv_host, size_t recv_count,
                        size_t type_size,
                        int peer_send, int peer_recv,
                        MPI_Comm comm);

void gpu_comm_send_recv_default(const void* send_host, size_t send_count,
                               void* recv_host, size_t recv_count,
                               size_t type_size,
                               int peer_send, int peer_recv,
                               MPI_Comm comm);

void* gpu_comm_default_recv_device_ptr();
