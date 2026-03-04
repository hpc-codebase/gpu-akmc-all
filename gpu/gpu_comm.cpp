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

#else

bool gpu_comm_enabled() { return false; }
void gpu_comm_init(MPI_Comm) {}
void gpu_comm_finalize() {}
void gpu_comm_sanity_allreduce(MPI_Comm) {}

#endif

