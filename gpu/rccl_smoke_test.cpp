#include "rccl_smoke_test.h"

#ifdef KMC_RCCL_ENABLE

#include <hip/hip_runtime.h>
#include <nccl.h>
#include <cstdlib>
#include <iostream>

static void check_nccl(ncclResult_t r, const char* file, int line) {
  if (r == ncclSuccess) return;
  std::cerr << "NCCL/RCCL error at " << file << ":" << line << " " << ncclGetErrorString(r) << std::endl;
  std::abort();
}

#define CHECK_NCCL(expr) check_nccl((expr), __FILE__, __LINE__)

static bool env_enabled(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

void rccl_smoke_test() {
  if (!env_enabled("KMC_RCCL_SMOKE")) return;

  int ver = 0;
  CHECK_NCCL(ncclGetVersion(&ver));
  std::cout << "[KMC_RCCL_SMOKE] nccl_version=" << ver << std::endl;

  ncclUniqueId id{};
  CHECK_NCCL(ncclGetUniqueId(&id));

  ncclComm_t comm{};
  CHECK_NCCL(ncclCommInitRank(&comm, 1, id, 0));

  hipStream_t stream{};
  hipError_t herr = hipStreamCreate(&stream);
  if (herr != hipSuccess) {
    std::cerr << "[KMC_RCCL_SMOKE] hipStreamCreate failed" << std::endl;
    std::abort();
  }

  float h_in = 3.25f;
  float h_out = 0.0f;
  float* d_in = nullptr;
  float* d_out = nullptr;
  herr = hipMalloc(&d_in, sizeof(float));
  if (herr != hipSuccess) std::abort();
  herr = hipMalloc(&d_out, sizeof(float));
  if (herr != hipSuccess) std::abort();
  herr = hipMemcpy(d_in, &h_in, sizeof(float), hipMemcpyHostToDevice);
  if (herr != hipSuccess) std::abort();

  CHECK_NCCL(ncclAllReduce(d_in, d_out, 1, ncclFloat, ncclSum, comm, stream));
  herr = hipStreamSynchronize(stream);
  if (herr != hipSuccess) std::abort();

  herr = hipMemcpy(&h_out, d_out, sizeof(float), hipMemcpyDeviceToHost);
  if (herr != hipSuccess) std::abort();

  std::cout << "[KMC_RCCL_SMOKE] allreduce_out=" << h_out << std::endl;

  hipFree(d_in);
  hipFree(d_out);
  hipStreamDestroy(stream);
  ncclCommDestroy(comm);
}

#else

void rccl_smoke_test() {}

#endif

