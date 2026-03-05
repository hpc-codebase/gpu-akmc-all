#include "gpu_comm.h"

#ifdef KMC_RCCL_ENABLE

#include <hip/hip_runtime.h>
#if __has_include(<nccl.h>)
#include <nccl.h>
#else
#include <rccl.h>
#endif

#include <cstdlib>
#include <iostream>

static bool env_enabled(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

static void check_nccl(ncclResult_t r, const char* file, int line) {
  if (r == ncclSuccess) return;
  std::cerr << "NCCL/RCCL error at " << file << ":" << line << " " << ncclGetErrorString(r) << std::endl;
  std::abort();
}

#define CHECK_NCCL(expr) check_nccl((expr), __FILE__, __LINE__)

static ncclComm_t g_comm{};
static hipStream_t g_stream{};
static bool g_inited = false;

bool gpu_comm_enabled() {
  return env_enabled("KMC_GPU_COMM");
}

void gpu_comm_init(MPI_Comm comm) {
  if (!gpu_comm_enabled()) return;
  if (g_inited) return;

  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(comm, &world_rank);
  MPI_Comm_size(comm, &world_size);

  ncclUniqueId id{};
  if (world_rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);

  CHECK_NCCL(ncclCommInitRank(&g_comm, world_size, id, world_rank));
  hipError_t herr = hipStreamCreate(&g_stream);
  if (herr != hipSuccess) std::abort();
  g_inited = true;
}

void gpu_comm_finalize() {
  if (!g_inited) return;
  hipStreamDestroy(g_stream);
  ncclCommDestroy(g_comm);
  g_inited = false;
}

void gpu_comm_sanity_allreduce(MPI_Comm comm) {
  if (!gpu_comm_enabled()) return;
  gpu_comm_init(comm);
  if (!g_inited) return;

  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(comm, &world_rank);
  MPI_Comm_size(comm, &world_size);

  int h_in = world_rank + 1;
  int h_out = 0;
  int* d_in = nullptr;
  int* d_out = nullptr;
  hipError_t herr = hipMalloc(&d_in, sizeof(int));
  if (herr != hipSuccess) std::abort();
  herr = hipMalloc(&d_out, sizeof(int));
  if (herr != hipSuccess) std::abort();
  herr = hipMemcpy(d_in, &h_in, sizeof(int), hipMemcpyHostToDevice);
  if (herr != hipSuccess) std::abort();

  CHECK_NCCL(ncclAllReduce(d_in, d_out, 1, ncclInt, ncclSum, g_comm, g_stream));
  herr = hipStreamSynchronize(g_stream);
  if (herr != hipSuccess) std::abort();
  herr = hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost);
  if (herr != hipSuccess) std::abort();

  hipFree(d_in);
  hipFree(d_out);

  if (world_rank == 0) {
    const int expected = (world_size * (world_size + 1)) / 2;
    std::cout << "[KMC_GPU_COMM] sanity_allreduce_out=" << h_out << " expected=" << expected << std::endl;
  }
}

void gpu_comm_buffer_free(GpuCommBuffer& buf) {
  if (buf.d_send) { hipFree(buf.d_send); buf.d_send = nullptr; }
  if (buf.d_recv) { hipFree(buf.d_recv); buf.d_recv = nullptr; }
  buf.send_cap = 0;
  buf.recv_cap = 0;
}

void gpu_comm_send_recv(GpuCommBuffer& buf,
                        const void* send_host, size_t send_count,
                        void* recv_host, size_t recv_count,
                        size_t type_size,
                        int peer_send, int peer_recv,
                        MPI_Comm comm) {
  if (!gpu_comm_enabled()) return;
  if (!g_inited) gpu_comm_init(comm);

  size_t send_bytes = send_count * type_size;
  size_t recv_bytes = recv_count * type_size;

  if (send_bytes > buf.send_cap) {
    if (buf.d_send) hipFree(buf.d_send);
    hipError_t err = hipMalloc(&buf.d_send, send_bytes);
    if (err != hipSuccess) {
      std::cerr << "HIP malloc failed (send): " << hipGetErrorString(err) << std::endl;
      std::abort();
    }
    buf.send_cap = send_bytes;
  }

  if (recv_bytes > buf.recv_cap) {
    if (buf.d_recv) hipFree(buf.d_recv);
    hipError_t err = hipMalloc(&buf.d_recv, recv_bytes);
    if (err != hipSuccess) {
      std::cerr << "HIP malloc failed (recv): " << hipGetErrorString(err) << std::endl;
      std::abort();
    }
    buf.recv_cap = recv_bytes;
  }

  if (send_bytes > 0) {
    hipError_t err = hipMemcpyAsync(buf.d_send, send_host, send_bytes, hipMemcpyHostToDevice, g_stream);
    if (err != hipSuccess) {
      std::cerr << "HIP memcpy H2D failed: " << hipGetErrorString(err) << std::endl;
      std::abort();
    }
  }

  CHECK_NCCL(ncclGroupStart());
  if (send_bytes > 0) {
    CHECK_NCCL(ncclSend(buf.d_send, send_bytes, ncclChar, peer_send, g_comm, g_stream));
  }
  if (recv_bytes > 0) {
    CHECK_NCCL(ncclRecv(buf.d_recv, recv_bytes, ncclChar, peer_recv, g_comm, g_stream));
  }
  CHECK_NCCL(ncclGroupEnd());

  if (recv_bytes > 0) {
    hipError_t err = hipMemcpyAsync(recv_host, buf.d_recv, recv_bytes, hipMemcpyDeviceToHost, g_stream);
    if (err != hipSuccess) {
      std::cerr << "HIP memcpy D2H failed: " << hipGetErrorString(err) << std::endl;
      std::abort();
    }
  }

  hipError_t err = hipStreamSynchronize(g_stream);
  if (err != hipSuccess) {
    std::cerr << "HIP stream sync failed: " << hipGetErrorString(err) << std::endl;
    std::abort();
  }
}

#else

bool gpu_comm_enabled() { return false; }
void gpu_comm_init(MPI_Comm) {}
void gpu_comm_finalize() {}
void gpu_comm_sanity_allreduce(MPI_Comm) {}
void gpu_comm_buffer_free(GpuCommBuffer&) {}
void gpu_comm_send_recv(GpuCommBuffer&, const void*, size_t, void*, size_t, size_t, int, int, MPI_Comm) {}

#endif
