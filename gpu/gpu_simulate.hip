
#include "gpu_simulate.h"
#include <cstdlib>
#include <mpi.h>

// #include <hiprand_kernel.h>
#ifndef KMC_RECB_DEVICE_PRINTF
#define KMC_RECB_DEVICE_PRINTF 0
#endif

// #define const int MAX_PROBE_STEPS = 3

// Global Variables
__device__ static GPUHashSet *g_MoRe_Hash;
__device__ static GPUHashSet *g_MoMo_Hash;
__device__ static GPUHashSet *g_Re_Hash;
__device__ static GPUHashSet *g_V_Hash;
__device__ static dev_meta *g_meta;
__device__ static GPUHashSet *g_Busy_Set;
__device__ static GPUHashSet *g_surface_Hash;
__device__ static GPUHashSet *g_ghost_Hash;
__device__ static GPUHashSet *g_comm_ghost_bucket0;
__device__ static GPUHashSet *g_comm_ghost_bucket1;
__device__ static GPUHashSet *g_comm_ghost_bucket2;
__device__ static GPUHashSet *g_comm_ghost_bucket3;
__device__ static GPUHashSet *g_comm_ghost_bucket4;
__device__ static GPUHashSet *g_comm_ghost_bucket5;
__device__ static GPUHashSet *g_comm_ghost_bucket6;
__device__ static comm::Region<comm::_type_lattice_size> *g_comm_ghost_regions;
__device__ static GPUHashSet **g_comm_surface_buckets_ptr;
__device__ static CommBox *g_comm_surface_boxes;
__device__ static int *g_comm_surface_offsets;
__device__ static long int  *g_Pair_Atoms;
__device__ unsigned long long int* g_Global_Bitmap_Lock;//哈希表的全局位图锁

static long int arr_size;
static long int pair_atoms_size;

struct RecbVerifyStats {
  unsigned long long hops;
  unsigned int max_hops;
  unsigned int multi_hop_pairs;
  unsigned int captured_no_move;
  unsigned int sector_pairs;
};

namespace cg = cooperative_groups;
//这些都是GPU内部的全局变量
namespace gpuSimulate {

    // device variable
    double *d_sum_rates;
    double *d_config;
    dev_event *d_event;

    dev_Vacancy *deviceVacancyInfo;
    dev_nnLattice *deviceNeighbourInfo;

    // host variable
    _type_lattice_count currentVacCount;
    double h_config[2];

    dev_atom *atoms;
    dev_atom *d_atoms;
    dev_atom *ret_atoms;
    dev_Vacancy *hostVacancyInfo;
    dev_nnLattice *hostNeighbourInfo;

    int blockSize;
    int numBlocks;
    _type_lattice_count vacHostArrayCapacity;
    _type_lattice_count HostAtomCapacity;

    long int *d_vac_all;
    long int *d_vac_sector;
    int *d_vac_count_all;
    int *d_vac_count_sector;
    int vac_capacity_all;
    int vac_capacity_sector;

    double *sum_rates;
    dev_event* h_event;

    _type_lattice_id to_x;
    _type_lattice_id to_y;
    _type_lattice_id to_z;

    LatticesList *lattice_list;
    comm::ColoredDomain *p_domain;


    //下面的是recb的命名空间

    dev_meta meta;
    HIPHashSet *MoRe_Hash;
    HIPHashSet *MoMo_Hash;
    HIPHashSet *Re_Hash;
    HIPHashSet *V_Hash;
    HIPHashSet *Busy_Set;
    HIPHashSet *ghost_Hash;
    HIPHashSet *surface_Hash;
    HIPHashSet *comm_ghost_bucket0;
    HIPHashSet *comm_ghost_bucket1;
    HIPHashSet *comm_ghost_bucket2;
    HIPHashSet *comm_ghost_bucket3;
    HIPHashSet *comm_ghost_bucket4;
    HIPHashSet *comm_ghost_bucket5;
    HIPHashSet *comm_ghost_bucket6;
    comm::Region<comm::_type_lattice_size> *d_comm_ghost_regions;
    int comm_ghost_capacity;
    HIPHashSet *comm_surface_buckets[24];
    GPUHashSet **d_comm_surface_bucket_ptrs;
    CommBox *d_comm_surface_boxes;
    int *d_comm_surface_offsets;
    int comm_surface_capacity;
    int comm_surface_boxes_cap;
    int comm_surface_offsets_cap;

    long int *Pair_Atoms;
    long int *d_Pair_Atoms;
    dev_meta *d_meta;
    RecbVerifyStats *d_recb_verify_stats = nullptr;

    unsigned long long int* d_Global_Bitmap_Lock = nullptr;
    size_t bitmap_lock_bytes = 0;
    GpuCommMaintenanceProfiler comm_gpu_maintenance_profiler;

    long int *d_ghost_list;
    long int *d_surface_list;
    long int *d_ghost_list_type;
    long int *d_surface_list_type;
    int *d_ghost_count;
    int *d_surface_count;
    int ghost_capacity;
    int surface_capacity;

    static bool recbProfileEnabled() {
      static int cached = -1;
      if (cached != -1) return cached != 0;
      const char* v = std::getenv("KMC_RECB_PROFILE");
      cached = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
      return cached != 0;
    }

    static hipEvent_t recb_evt_begin{};
    static hipEvent_t recb_evt_after_exchange{};
    static hipEvent_t recb_evt_after_type{};
    static hipEvent_t recb_evt_after_copy{};
    static hipEvent_t recb_evt_after_clear{};
    static bool recb_evt_inited = false;

    static float recb_ms_exchange = 0.0f;
    static float recb_ms_type = 0.0f;
    static float recb_ms_copy = 0.0f;
    static float recb_ms_clear = 0.0f;

    static void initRecbProfileEvents() {
      if (recb_evt_inited) return;
      HANDLE_HIP(hipEventCreate(&recb_evt_begin));
      HANDLE_HIP(hipEventCreate(&recb_evt_after_exchange));
      HANDLE_HIP(hipEventCreate(&recb_evt_after_type));
      HANDLE_HIP(hipEventCreate(&recb_evt_after_copy));
      HANDLE_HIP(hipEventCreate(&recb_evt_after_clear));
      recb_evt_inited = true;
    }

    static void destroyRecbProfileEvents() {
      if (!recb_evt_inited) return;
      hipEventDestroy(recb_evt_begin);
      hipEventDestroy(recb_evt_after_exchange);
      hipEventDestroy(recb_evt_after_type);
      hipEventDestroy(recb_evt_after_copy);
      hipEventDestroy(recb_evt_after_clear);
      recb_evt_inited = false;
    }
}

void resetGpuCommMaintenanceProfiler() {
    gpuSimulate::comm_gpu_maintenance_profiler = GpuCommMaintenanceProfiler{};
}

GpuCommMaintenanceProfiler getGpuCommMaintenanceProfiler() {
    return gpuSimulate::comm_gpu_maintenance_profiler;
}

void gpuCommProfileAddPack(double seconds) {
    gpuSimulate::comm_gpu_maintenance_profiler.pack_time += seconds;
}

void gpuCommProfileAddNccl(double seconds) {
    gpuSimulate::comm_gpu_maintenance_profiler.nccl_time += seconds;
}

void gpuCommProfileAddHashOp(double seconds) {
    gpuSimulate::comm_gpu_maintenance_profiler.hash_op_time += seconds;
}

void gpuCommProfileAddCompactSample(double seconds, bool first_of_dim) {
    if (first_of_dim) {
        gpuSimulate::comm_gpu_maintenance_profiler.compact_first_sum += seconds;
        gpuSimulate::comm_gpu_maintenance_profiler.compact_first_n += 1.0;
    } else {
        gpuSimulate::comm_gpu_maintenance_profiler.compact_rest_sum += seconds;
        gpuSimulate::comm_gpu_maintenance_profiler.compact_rest_n += 1.0;
    }
}

__device__ _type_lattice_id getId(_type_lattice_size x,_type_lattice_size y,_type_lattice_size z);
__device__ int isGhostLat(_type_lattice_id lid);
__device__ int isSurfaceLat(_type_lattice_id lid);
static __device__ __forceinline__ GPUHashSet* commBucketPtr(int idx);

static int parseLocalRankEnv(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return -1;
  char* end = nullptr;
  const long rank = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || rank < 0) return -1;
  return static_cast<int>(rank);
}

// Node-local rank for GPU bind. MPI_COMM_TYPE_SHARED is used first so
// sbatch+mpirun cannot inherit a stale SLURM_LOCALID=0 onto remote nodes.
static int getLocalProcessRank(int fallback_rank) {
  int mpi_inited = 0;
  MPI_Initialized(&mpi_inited);
  if (mpi_inited) {
    MPI_Comm node_comm = MPI_COMM_NULL;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, fallback_rank,
                            MPI_INFO_NULL, &node_comm) == MPI_SUCCESS &&
        node_comm != MPI_COMM_NULL) {
      int local_rank = 0;
      MPI_Comm_rank(node_comm, &local_rank);
      MPI_Comm_free(&node_comm);
      return local_rank;
    }
  }

  const char* env_names[] = {
      "OMPI_COMM_WORLD_LOCAL_RANK",
      "MV2_COMM_WORLD_LOCAL_RANK",
      "MPI_LOCALRANKID",
      "PMI_LOCAL_RANK",
      "SLURM_LOCALID",
  };
  for (const char* name : env_names) {
    const int rank = parseLocalRankEnv(name);
    if (rank >= 0) return rank;
  }
  return fallback_rank;
}

static int getEnvIntOrDefault(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') return default_value;
  return static_cast<int>(parsed);
}

static bool shouldPrintGpuBindDebug(int process_rank, int local_rank) {
  if (std::getenv("KMC_GPU_BIND_DEBUG") == nullptr) return false;
  const int max_global = getEnvIntOrDefault("KMC_GPU_BIND_DEBUG_MAX_GLOBAL", 16);
  const int local_filter = getEnvIntOrDefault("KMC_GPU_BIND_DEBUG_LOCAL_RANK", -1);
  if (local_filter >= 0) return local_rank == local_filter;
  return process_rank < max_global;
}
// =================================================================================
//[防线 L1] 组内冲突拦截器 (Intra-Tile Resolvers)
// =================================================================================
// 针对双目标 (calcExchangePairs 用)
template <typename TileType>
__device__ void resolve_intra_tile_conflicts(AtomTx& tx, TileType tile) {
    int my_rank = tile.thread_rank();
    bool conflict = false;
    const bool me_can_conflict = tx.active && tx.op_type != 0;

    for (int i = 0; i < tile.size(); ++i) {
        long int other_src = (long int)tile.shfl((unsigned long long int)tx.src_id, i);
        long int other_dst = (long int)tile.shfl((unsigned long long int)tx.dst_id, i);
        bool other_active  = tile.shfl(tx.active, i);
        int other_op       = tile.shfl(tx.op_type, i);

        if (me_can_conflict && other_active && other_op != 0 && i != my_rank) {
             bool overlap = (tx.dst_id == other_dst) ||
                           (tx.dst_id == other_src) ||
                           (tx.src_id == other_dst) ||
                           (tx.src_id == other_src);
            if (overlap && my_rank > i) {
                conflict = true;
            }
        }
    }
    if (conflict) tx.op_type = 0;
}

// 针对单目标 (update_hash_set 用)
template <typename TileType>
__device__ void resolve_intra_tile_conflicts_single(long int id, bool& active, TileType tile) {
    if (!active) return;
    int my_rank = tile.thread_rank();
    bool conflict = false;

    for (int i = 0; i < tile.size(); ++i) {
        long int other_id = (long int)tile.shfl((unsigned long long int)id, i);
        bool other_active = tile.shfl(active, i);
        if (other_active && id == other_id && i != my_rank && my_rank > i) {
            conflict = true;
        }
    }
    if (conflict) active = false; 
}

// =================================================================================
// [防线 L2] 64 位全局位图锁 (Wavefront-Aligned Bitmap Locks)
// =================================================================================
// 针对双目标 (calcExchangePairs 用)
__device__ inline long int get_safe_lock_id(long int id) {
    long int total_size = g_meta->size_x * g_meta->size_y * g_meta->size_z;
    return (id % total_size + total_size) % total_size;
}
__device__ inline bool try_acquire_bitmap_locks(long int src_id, long int dst_id) {
    if (src_id == dst_id) return false;
    long int safe_src = get_safe_lock_id(src_id);
    long int safe_dst = get_safe_lock_id(dst_id);

    long int min_id = min(safe_src, safe_dst);
    long int max_id = max(safe_src, safe_dst);

    unsigned long long int idx_min = min_id / 64;
    unsigned long long int mask_min = 1ULL << (min_id % 64); // 必须用 1ULL!
    unsigned long long int idx_max = max_id / 64;
    unsigned long long int mask_max = 1ULL << (max_id % 64);

    if ((atomicOr(&g_Global_Bitmap_Lock[idx_min], mask_min) & mask_min) == 0) {
        if ((atomicOr(&g_Global_Bitmap_Lock[idx_max], mask_max) & mask_max) == 0) {
            return true; // 双锁达成
        } else {
            atomicAnd(&g_Global_Bitmap_Lock[idx_min], ~mask_min); // 回退第一把锁
            return false;
        }
    }
    return false;
}

__device__ inline void release_bitmap_locks(long int src_id, long int dst_id) {
    long int safe_src = get_safe_lock_id(src_id);
    long int safe_dst = get_safe_lock_id(dst_id);
    atomicAnd(&g_Global_Bitmap_Lock[safe_src / 64], ~(1ULL << (safe_src % 64)));
    atomicAnd(&g_Global_Bitmap_Lock[safe_dst / 64], ~(1ULL << (safe_dst % 64)));
}

// 针对单目标 (update_hash_set 用)
__device__ inline bool try_acquire_bitmap_lock_single(long int id) {
    unsigned long long int mask = 1ULL << (id % 64);
    return (atomicOr(&g_Global_Bitmap_Lock[id / 64], mask) & mask) == 0;
}

__device__ inline void release_bitmap_lock_single(long int id) {
    unsigned long long int mask = 1ULL << (id % 64);
    atomicAnd(&g_Global_Bitmap_Lock[id / 64], ~mask);
}
// =========================================================
// 单线程查找操作 (Single-Thread Contains)
// 完美兼容 Tile 协作组的桶状哈希表结构 (Bucketed Hash Map)
// 适用于未声明协作组的普通线程调用
// 返回值: true (找到), false (未找到)
// =========================================================
__device__ bool hash_set_contains(GPUHashSet* set, long int key) {
    unsigned int loop_count = 0;
    const unsigned int max_probe = hash_effective_max_probes(set->num_buckets);
    unsigned int buck_num = hash_probe_bucket(set->num_buckets, key, loop_count);
    bool result = false;
    // printf("debug_test_contain, buckets:%d capacity: %d\n", set->num_buckets,set->capacity);


    while (loop_count <= max_probe ) {
        int base_index = buck_num * BUCKET_SIZE;
        bool should_stop = false;

        for (int i = 0; i < BUCKET_SIZE; ++i) {
            // long int val = set->keys[base_index + i].load(hip::memory_order_relaxed);
            long int val = atomic_load_relaxed(&set->keys[base_index + i]);
            
            if (val == key) {
                result = true; 
                should_stop = true; 
                break;
            }
            if (val == set->empty_flag) {
                should_stop = true; // 遇到空位，彻底不存在
                break;              
            }
        }       
        if (should_stop) {
          break; 
        }

        loop_count++;
        buck_num = hash_probe_bucket(set->num_buckets, key, loop_count);
    }
    // printf("debug_test_contain, result:%d\n", result);
    return result;
}
//单线程插入
// 返回值: 1 (插入成功), 0 (已存在，未重复插入), -1 (表满失败)
__device__ int hash_set_insert_atom_op(GPUHashSet* set, long int key) {
    unsigned int loop_count = 0;
    const unsigned int max_probe = hash_effective_max_probes(set->num_buckets);
    unsigned int buck_num = hash_probe_bucket(set->num_buckets, key, loop_count);
    // const int MAX_PROBE_STEPS = 3;
    while (loop_count <= max_probe) {
        int base_index = buck_num * BUCKET_SIZE;

        // 在桶内的 64 个槽位中线性查找
        for (int i = 0; i < BUCKET_SIZE; ++i) {
            long int* target_addr = &set->keys[base_index + i];
            long int val = atomic_load_relaxed(target_addr);
            if (val == key) return 0;
       
            if (val == set->empty_flag || val == set->tombstone_flag) {
                unsigned long long int old_val = atomicCAS(
                    (unsigned long long int*)target_addr, 
                    (unsigned long long int)val, 
                    (unsigned long long int)key
                );

                if (old_val == (unsigned long long int)val) return 1;
                i--; 
                continue;
            }
        }
        
        // 当前桶满了，去下一个桶
        loop_count++;
        buck_num = hash_probe_bucket(set->num_buckets, key, loop_count);
    }
    return -1;
}
//单线程删除
// 返回值: 1 (删除成功), 0 (未找到该 Key)
__device__ int hash_set_remove_atom_op(GPUHashSet* set, long int key) {
    unsigned int loop_count = 0;
    const unsigned int max_probe = hash_effective_max_probes(set->num_buckets);
    unsigned int buck_num = hash_probe_bucket(set->num_buckets, key, loop_count);

    while (loop_count <= max_probe) {
        int base_index = buck_num * BUCKET_SIZE;

        for (int i = 0; i < BUCKET_SIZE; ++i) {
            long int* target_addr = &set->keys[base_index + i];
            long int val = atomic_load_relaxed(target_addr);
            
            // 情况 1: 找到了目标 Key
            if (val == key) {
                // 将其原子替换为墓碑标记
                unsigned long long int old_val = atomicCAS(
                    (unsigned long long int*)target_addr, 
                    (unsigned long long int)key, 
                    (unsigned long long int)set->tombstone_flag
                );

                // 抢占成功
                if (old_val == (unsigned long long int)key) return 1;

                // 抢占失败（说明就在这一瞬间，这个 Key 刚好被别的逻辑删了或改了）
                i--; // 重新检查
                continue;
            }

            // 情况 2: 遇到空位，说明哈希表后面不可能有这个 Key 了
            if (val == set->empty_flag) return 0;
        }

        loop_count++;
        buck_num = hash_probe_bucket(set->num_buckets, key, loop_count);
    }
    return 0;
}


template <typename TileType>
__device__ __noinline__ int tiled_contains_op(GPUHashSet* set, long int key, TileType tile, bool active) {
    int final_result = 0;
    uint64_t work_mask = tile.ballot(active);

    while (work_mask != 0) {
        int leader = __ffsll((unsigned long long int)work_mask) - 1;
        long int leader_key = tile.shfl(key, leader); 
        
        int lane = tile.thread_rank();

        int loop_count = 0;
        unsigned int buck_num = hash_probe_bucket(set->num_buckets, leader_key, loop_count);
        int leader_result = 0;
        int searching = 1;

        // 【日志】：Leader 宣布任务开始
        // if (lane == leader) {
        //     printf("  -> [Contains] Leader %d starting search for Key %ld at Bucket %u\n", leader, leader_key, buck_num);
        // }

        while(searching) {
            // long int val = set->keys[buck_num * 64 + lane].load(hip::memory_order_relaxed);
            long int val = atomic_load_relaxed(&set->keys[buck_num * BUCKET_SIZE + lane]);
            
            uint64_t match_mask = tile.ballot(val == leader_key);
            uint64_t empty_mask = tile.ballot(val == set->empty_flag);

            if (match_mask != 0) {
                leader_result = 1; 
                searching = 0; 
                // if (lane == leader) printf("    -> [HIT] Key %ld found at bucket %u by mask %llx\n", leader_key, buck_num, match_mask);
            } else if (empty_mask != 0) {
                leader_result = 0; 
                searching = 0; 
                // if (lane == leader) printf("    -> [MISS] Encountered empty slots at bucket %u. Key %ld does not exist.\n", buck_num, leader_key);
            } else {
                loop_count++;
                buck_num = hash_probe_bucket(set->num_buckets, leader_key, loop_count);
                // if (lane == leader) printf("    -> [COLLISION] Bucket full, probing next bucket %u (Loop %d)\n", buck_num, loop_count);

                if (loop_count > set->num_buckets ) {  
                    leader_result = 0; 
                    searching = 0; 
                    // if (lane == leader) printf("    -> [FAIL] Hash table fully probed. Key %ld not found.\n", leader_key);
                }
            }
        }

        if (lane == leader) final_result = leader_result;
        work_mask &= ~(1ULL << leader);
    }
    return final_result;
}

// ---------------------------------------------------------
// 2. Tiled Insert (插入操作)
// 返回值: 1 插入成功, -1 已经存在(跳过), -2 表满失败
// ---------------------------------------------------------
template <typename TileType>
__device__ int tiled_insert_op(GPUHashSet* set, long int key, TileType tile, bool active, int max_probes) {
    int final_result = 0; 
    const unsigned int probe_limit = hash_effective_max_probes(set->num_buckets);
    const unsigned int effective_max_probes =
        max_probes < static_cast<int>(probe_limit) ? static_cast<unsigned int>(max_probes) : probe_limit;
    uint64_t work_mask = tile.ballot(active);

    while (work_mask != 0) {
        int leader = __ffsll((unsigned long long int)work_mask) - 1;
        // long int leader_key = tile.shfl(key, leader);
        long int leader_key = (long int)tile.shfl((unsigned long long int)key, leader);
        
        int lane = tile.thread_rank();
        int loop_count = 0;
        unsigned int buck_num = hash_probe_bucket(set->num_buckets, leader_key, loop_count);
        int leader_result = 0;
        // if (lane == leader) {
        //     printf("  -> [Insert] Leader %d starting search for Key %ld at Bucket %u\n", leader, leader_key, buck_num);
        // }
        int searching = 1;
        while(searching){
          // long int val = set->keys[buck_num * 64 + lane].load(hip::memory_order_relaxed);
          long int val = atomic_load_relaxed(&set->keys[buck_num * BUCKET_SIZE + lane]);
          if (tile.ballot(val == leader_key) != 0) {
              leader_result = 0;
              searching = 0;
          } else {

          // // 1. 插入之前检查当前key是否已经存在
          // if (tile.ballot(val == leader_key) != 0) {
          //     leader_result = -1; // 已存在
          //     searching = 0;
          // }else{
            // 2. 寻找空位或墓碑
            uint64_t candidate_mask = tile.ballot(val == set->empty_flag) |
                                      tile.ballot(val == set->tombstone_flag);
            
            if (candidate_mask != 0) {
                int target_lane = __ffsll((unsigned long long int)candidate_mask) - 1;
                 int cas_success = 0; 
                    
                if (lane == target_lane) {
                    // #if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
                    // long int expected = val;
                    // cas_success = set->keys[buck_num * 64 + lane].compare_exchange_strong(
                    //         expected, leader_key, 
                    //         hip::memory_order_seq_cst, hip::memory_order_seq_cst);
                    unsigned long long int* addr = (unsigned long long int*)&set->keys[buck_num * BUCKET_SIZE + lane];
                    unsigned long long int old_val = atomicCAS(addr, (unsigned long long int)val, (unsigned long long int)leader_key);
                    cas_success = (old_val == (unsigned long long int)val) ? 1 : 0;
                    // #endif
                    // cas_success = do_safe_atomic_cas((long int*)&set->keys[buck_num * 64 + lane], val, leader_key);
                }
                
                if (tile.ballot(cas_success!=0) != 0) {
                    leader_result = 1; 
                    searching = 0;
                }
                // 如果插入失败(被同一Wave内其他操作抢占)，继续在当前桶内重试
            } else {
                // 桶已满，线性探测到下一个桶
                loop_count++;
                buck_num = hash_probe_bucket(set->num_buckets, leader_key, loop_count);
                if (loop_count > effective_max_probes) {
                    leader_result = -2; // 插入失败(表满)
                    searching = 0;
                }
            }
          }
        }

        if (lane == leader) final_result = leader_result;
        work_mask &= ~(1ULL << leader);
    }
    return final_result;
}

// ---------------------------------------------------------
// 3. Tiled Remove (删除操作)
// 返回值: 1 删除成功, 0 未找到或已被别人删除
// ---------------------------------------------------------
template <typename TileType>
__device__ int tiled_remove_op(GPUHashSet* set, long int key, TileType tile, bool active, int max_probes) {
    int final_result = 0;
    const unsigned int probe_limit = hash_effective_max_probes(set->num_buckets);
    const unsigned int effective_max_probes =
        max_probes < static_cast<int>(probe_limit) ? static_cast<unsigned int>(max_probes) : probe_limit;
    uint64_t work_mask = tile.ballot(active); 
    while (work_mask != 0) {
        int leader = __ffsll((unsigned long long int)work_mask) - 1;
        // long int leader_key = tile.shfl(key, leader);
        long int leader_key = (long int)tile.shfl((unsigned long long int)key, leader);
        
        int lane = tile.thread_rank();
        int loop_count = 0;
        unsigned int buck_num = hash_probe_bucket(set->num_buckets, leader_key, loop_count);
        int leader_result = 0;
        // if (lane == leader) {
        //     printf("  -> [del] Leader %d starting search for Key %ld at Bucket %u\n", leader, leader_key, buck_num);
        // }
        int searching = 1;
        while(searching){
            // long int val = set->keys[buck_num * 64 + lane].load(hip::memory_order_relaxed);
            long int val = atomic_load_relaxed(&set->keys[buck_num * BUCKET_SIZE + lane]);
            uint64_t match_mask = tile.ballot(val == leader_key);
            
            if (match_mask != 0) {
                int target_lane = __ffsll((unsigned long long int)match_mask) - 1;
                int cas_success = 0;
                
                if (lane == target_lane) {
                    // #if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
                    // long int expected = leader_key;
                    // cas_success = set->keys[buck_num * 64 + lane].compare_exchange_strong(
                    //         expected, set->tombstone_flag, 
                    //         hip::memory_order_seq_cst, hip::memory_order_seq_cst);
                    unsigned long long int* addr = (unsigned long long int*)&set->keys[buck_num * BUCKET_SIZE + lane];
                    unsigned long long int old_val = atomicCAS(addr, (unsigned long long int)leader_key, (unsigned long long int)set->tombstone_flag);
                    cas_success = (old_val == (unsigned long long int)leader_key) ? 1 : 0;
                    // #endif
                    // cas_success = do_safe_atomic_cas((long int*)&set->keys[buck_num * 64 + lane], leader_key, set->tombstone_flag);
                }
                
                // 用 ballot 判定是否至少一个线程成功删除了
                leader_result = (tile.ballot(cas_success!=0) != 0) ? 1 : 0;
                searching = 0;
            } else if (tile.ballot(val == set->empty_flag) != 0) {
                leader_result = 0; // 未找到
                searching = 0;
            }else{
              loop_count++;
              buck_num = hash_probe_bucket(set->num_buckets, leader_key, loop_count);
              if (loop_count > effective_max_probes) {
                  leader_result = 0; // 未找到 (表满)
                  searching = 0;
              }
            }
        }

        if (lane == leader) final_result = leader_result;
        work_mask &= ~(1ULL << leader);
    }
    return final_result;
}


__device__ void printHashTableKernel(GPUHashSet* table, int max_entries, bool verbose) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {  // 仅由第一个线程执行
        printf("=== GPU Hash Table Info ===\n");
        printf("Capacity: %d\n", table->capacity);
        printf("Empty Flag: %ld\n", table->empty_flag);
        printf("Tombstone Flag: %ld\n", table->tombstone_flag);
        printf("---------------------------\n");
        
        const int display_count = min(max_entries, table->capacity);
        int valid_count = 0;
        
        for (int i = 0; i < display_count; i++) {
            long int key = table->keys[i];
            
            if (key == table->empty_flag) {
                if (verbose) printf("[%3d]: EMPTY\n", i);
            } else if (key == table->tombstone_flag) {
                if (verbose) printf("[%3d]: TOMBSTONE\n", i);
            } else {
                printf("[%3d]: %ld\n", i, key);
                valid_count++;
            }
        }
        
        if (table->capacity > display_count) {
            printf("... and %d more entries.\n", table->capacity - display_count);
        }
        
        float load_factor = (float)valid_count / table->capacity;
        printf("Total valid entries: %d\n", valid_count);
        printf("Load factor: %.3f\n", load_factor);
        printf("===========================\n");
    }
}
__global__ void print_pair_atoms(int arr_size){
  printf("===========start print pair==========");
  for(int i=0;i<arr_size * 2;i++){
    printf("%lld\t",g_Pair_Atoms[i]);
    if(i%10 == 0)
      printf("\n");
  }
  
  printf("===========end print pair============");
}
__device__ void printDevMetaKernel(const dev_meta* meta) {
    // 仅由第一个线程执行打印，避免输出重复
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("=== dev_meta Structure Information ===\n");
        
        // 打印晶格尺寸信息
        printf("Lattice Sizes:\n");
        printf("  size_x: %lld (simulation lattice size)\n", meta->size_x);
        printf("  size_y: %lld\n", meta->size_y);
        printf("  size_z: %lld\n", meta->size_z);
        
        // 打印当前进程的模拟盒尺寸
        printf("\nCurrent Process Box Sizes:\n");
        printf("  box_x: %lld (2x real size for BCC structure)\n", meta->box_x);
        printf("  box_y: %lld (same as real size)\n", meta->box_y);
        printf("  box_z: %lld (same as real size)\n", meta->box_z);
        
        // 打印全局模拟盒尺寸
        printf("\nGlobal Simulation Box Sizes:\n");
        printf("  g_box_x: %lld\n", meta->g_box_x);
        printf("  g_box_y: %lld\n", meta->g_box_y);
        printf("  g_box_z: %lld\n", meta->g_box_z);
        
        // 打印全局坐标基准
        printf("\nGlobal Coordinate Baselines:\n");
        printf("  g_base_x: %lld (for global lattice ID calculation)\n", meta->g_base_x);
        printf("  g_base_y: %lld\n", meta->g_base_y);
        printf("  g_base_z: %lld\n", meta->g_base_z);
        
        // 打印幽灵晶格尺寸
        printf("\nGhost Lattice Sizes:\n");
        printf("  ghost_x: %lld (2x real size)\n", meta->ghost_x);
        printf("  ghost_y: %lld\n", meta->ghost_y);
        printf("  ghost_z: %lld\n", meta->ghost_z);
        
        // 打印ID相关信息
        printf("\nID Information:\n");
        printf("  _max_id: %d (max local lattice ID)\n", meta->_max_id);
        printf("  local_base_id: %d (global ID = local ID + this value)\n", meta->local_base_id);
        
        // 打印regions数组（分维度显示）
        printf("\nRegions Array (48 elements):\n");
        const int regions_per_dim = 8;
        for (int dim = 0; dim < 3; ++dim) {
            printf("  Dimension %d Regions:\n", dim);
            for (int region = 0; region < regions_per_dim; ++region) {
                int index = dim * regions_per_dim + region;
                printf("    Region %d: %lld", region, meta->regions[index]);
                if (region % 2 == 1) {
                    printf(" (boundary: %lld to %lld)", meta->regions[index - 1], meta->regions[index]);
                }
                printf("\n");
            }
        }
        
        printf("=====================================\n");
    }
}
__device__ void printTopN(GPUHashSet* table, int max_entries){
   if (threadIdx.x == 0 && blockIdx.x == 0) {  // 仅由第一个线程执行
        printf("=== TOP %d info===\n",max_entries);
        printf("---------------------------\n");
        
        const int display_count = min(max_entries, table->capacity);
        int valid_count = 0;
        
        for (int i = 0; i < table->capacity; i++) {
            long int key = table->keys[i];
            if(valid_count >= max_entries)
              break;
            if (key == table->empty_flag) {
                continue;
            } else if (key == table->tombstone_flag) {
                continue;
            } else {
                printf("[%3d]: %ld\t", i, key);
                valid_count++;
            }
        }
        printf("\n");
      
        printf("===========================\n");
    }
}
__global__ void print_g_Pair_Atoms(int arr_size){
  printf("==========pair_atoms============\n");
  for(int i = 0;i<arr_size;i++){
    printf("%ld\t",g_Pair_Atoms[i]);
    if(i%5==0)
      printf("\n");
  }
  printf("\n");
  printf("================================\n");
}
void printDevMeta(const dev_meta& meta) {
    std::cout << "=== dev_meta Structure Information ===" << std::endl;
    
    // 打印晶格尺寸信息
    std::cout << "Lattice Sizes:" << std::endl;
    std::cout << "  size_x: " << meta.size_x << " (simulation lattice size)" << std::endl;
    std::cout << "  size_y: " << meta.size_y << std::endl;
    std::cout << "  size_z: " << meta.size_z << std::endl;
    
    // 打印当前进程的模拟盒尺寸（注意box_x的特殊说明）
    std::cout << "\nCurrent Process Box Sizes:" << std::endl;
    std::cout << "  box_x: " << meta.box_x << " (2x real size for BCC structure)" << std::endl;
    std::cout << "  box_y: " << meta.box_y << " (same as real size)" << std::endl;
    std::cout << "  box_z: " << meta.box_z << " (same as real size)" << std::endl;
    
    // 打印全局模拟盒尺寸
    std::cout << "\nGlobal Simulation Box Sizes:" << std::endl;
    std::cout << "  g_box_x: " << meta.g_box_x << std::endl;
    std::cout << "  g_box_y: " << meta.g_box_y << std::endl;
    std::cout << "  g_box_z: " << meta.g_box_z << std::endl;
    
    // 打印全局坐标基准
    std::cout << "\nGlobal Coordinate Baselines:" << std::endl;
    std::cout << "  g_base_x: " << meta.g_base_x << " (for global lattice ID calculation)" << std::endl;
    std::cout << "  g_base_y: " << meta.g_base_y << std::endl;
    std::cout << "  g_base_z: " << meta.g_base_z << std::endl;
    
    // 打印幽灵晶格尺寸
    std::cout << "\nGhost Lattice Sizes:" << std::endl;
    std::cout << "  ghost_x: " << meta.ghost_x << " (2x real size)" << std::endl;
    std::cout << "  ghost_y: " << meta.ghost_y << std::endl;
    std::cout << "  ghost_z: " << meta.ghost_z << std::endl;
    
    // 打印ID相关信息
    std::cout << "\nID Information:" << std::endl;
    std::cout << "  _max_id: " << meta._max_id << " (max local lattice ID)" << std::endl;
    std::cout << "  local_base_id: " << meta.local_base_id << " (global ID = local ID + this value)" << std::endl;
    
    // 打印regions数组（分维度显示）
    std::cout << "\nRegions Array (48 elements):" << std::endl;
    const int regions_per_dim = 8;  // 假设每个维度有8个区域
    for (int dim = 0; dim < 3; ++dim) {
        std::cout << "  Dimension " << dim << " Regions:" << std::endl;
        for (int region = 0; region < regions_per_dim; ++region) {
            int index = dim * regions_per_dim + region;
            std::cout << "    Region " << region << ": " << meta.regions[index];
            if (region % 2 == 1) {
                std::cout << " (boundary: " << meta.regions[index - 1] << " to " << meta.regions[index] << ")";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "=====================================" << std::endl;
}
__device__ void checkpointers() {
    // 检查g_MoRe_Hash指针
    if (g_MoRe_Hash == nullptr) {
        printf("ERROR: g_MoRe_Hash is NULL!\n");
    } else {
        printf("g_MoRe_Hash is valid, add:%p capacity: %d\n", g_MoRe_Hash,g_MoRe_Hash->capacity);
    }
    
    // 检查g_MoMo_Hash指针
    if (g_MoMo_Hash == nullptr) {
        printf("ERROR: g_MoMo_Hash is NULL!\n");
    } else {
        printf("g_MoMo_Hash is valid, , add:%p,capacity: %d\n", g_MoMo_Hash,g_MoMo_Hash->capacity);
    }
    
    // 检查g_Re_Hash指针
    if (g_Re_Hash == nullptr) {
        printf("ERROR: g_Re_Hash is NULL!\n");
    } else {
        printf("g_Re_Hash is valid, capacity: %d\n", g_Re_Hash->capacity);
    }
    
    // 检查g_V_Hash指针
    if (g_V_Hash == nullptr) {
        printf("ERROR: g_V_Hash is NULL!\n");
    } else {
        printf("g_V_Hash is valid, capacity: %d\n", g_V_Hash->capacity);
    }
    
    // 检查g_meta指针
    if (g_meta == nullptr) {
        printf("ERROR: g_meta is NULL!\n");
    } else {
        printf("g_meta is valid, size_x: %lld\n", g_meta->size_x);
    }
    
    // 检查g_Busy_Set指针
    if (g_Busy_Set == nullptr) {
        printf("ERROR: g_Busy_Set is NULL!\n");
    } else {
        printf("g_Busy_Set is valid, capacity: %d\n", g_Busy_Set->capacity);
    }
    
    // 检查g_surface_Hash指针
    if (g_surface_Hash == nullptr) {
        printf("ERROR: g_surface_Hash is NULL!\n");
    } else {
        printf("g_surface_Hash is valid, capacity: %d\n", g_surface_Hash->capacity);
    }
    
    // 检查g_ghost_Hash指针
    if (g_ghost_Hash == nullptr) {
        printf("ERROR: g_ghost_Hash is NULL!\n");
    } else {
        printf("g_ghost_Hash is valid, capacity: %d\n", g_ghost_Hash->capacity);
    }
    
    // 检查g_Pair_Atoms指针
    if (g_Pair_Atoms == nullptr) {
        printf("ERROR: g_Pair_Atoms is NULL!\n");
    } else {
        printf("g_Pair_Atoms is valid, address: 0x%p\n", (void*)g_Pair_Atoms);
    }
}

__global__ void print_surface_and_ghost(){
  int g=0,s=0;
  long int x=0,y=0,z=0;
 

  printf("++++++++++++s+++++++++++++++++++++\n");
  for(int i=0;i<g_surface_Hash->capacity;i++)
    if(g_surface_Hash->keys[i]!=-1 && g_surface_Hash->keys[i]!=-1){
      long int id = g_surface_Hash->keys[i];
      x = id%g_meta->size_x;
      y = (id / g_meta->size_x) % g_meta->size_y;
      z = id / (g_meta->size_x * g_meta->size_y);
      printf("s(%ld,%ld)-(%ld,%ld,%ld)\t",g_surface_Hash->keys[i],getType(g_surface_Hash->keys[i]),x,y,z);
      s++;
      if(s%5==0)
      printf("\n");
    }
  printf("\n");
  printf("sssssssssssssss\n");
  for(int i=0;i<g_ghost_Hash->capacity;i++)
    if(g_ghost_Hash->keys[i]!=-1 && g_ghost_Hash->keys[i]!=-1){
      long int id = g_ghost_Hash->keys[i];
      x = id%g_meta->size_x;
      y = (id / g_meta->size_x) % g_meta->size_y;
      z = id / (g_meta->size_x * g_meta->size_y);
      printf("g(%ld,%ld)-(%ld,%ld,%ld)\t",g_ghost_Hash->keys[i],getType(g_ghost_Hash->keys[i]),x,y,z);
      g++;
      if(g%5==0)
      printf("\n");
    }
    printf("\n");
}
__global__ void print_Hash_and_Pair(int arr_size,int flag){
  long int pair_cnt=0,momo=0,more=0;
  for(int i=0;i<arr_size * 2;i++)
    if(g_Pair_Atoms[i]!=-1)
      pair_cnt++;
  for(int i=0;i<g_MoMo_Hash->capacity;i++)
    if(g_MoMo_Hash->keys[i]!=-1 && g_MoMo_Hash->keys[i]!=-2)
      momo++;
  for(int i=0;i<g_MoRe_Hash->capacity;i++)
    if(g_MoRe_Hash->keys[i]!=-1 && g_MoRe_Hash->keys[i]!=-2)
      more++;

  printf("==current hash and pairs========\n");
  if(flag == 1)
    printf("after:");
  printf("arr_size:%ld,pair: %ld\t momo %ld\t more %ld\n",arr_size,pair_cnt,momo,more);
  printf("==============================\n");
}
__global__ void print_momo_and_more(){
   printf("----------------------after momo-----------------------\n");
  for(int i=0;i<g_MoMo_Hash->capacity;i++)
    if(g_MoMo_Hash->keys[i]!=-1 && g_MoMo_Hash->keys[i]!=-2)
      printf("%ld\t",g_MoMo_Hash->keys[i]);
  printf("\n");
  printf("----------------------after more-----------------------\n");
  for(int i=0;i<g_MoRe_Hash->capacity;i++)
    if(g_MoRe_Hash->keys[i]!=-1 && g_MoRe_Hash->keys[i]!=-2)
      printf("%ld\t",g_MoRe_Hash->keys[i]);
  printf("\n");
}
__device__ double _1nnMapToSwitch(bonds::PairBond::bond_type bondType){
    switch (bondType) {
      case bonds::PairBond::MoMo:
        return -1.991;
      case bonds::PairBond::VV:
        return 0.199;
      case bonds::PairBond::ReRe:
        return -2.201;
      case bonds::PairBond::NiNi:
        return -1.618;
      case bonds::PairBond::MnMn:
        return -0.271;
      case bonds::PairBond::SiSi:
        return -0.716;
      case bonds::PairBond::VMo:
        return -0.755318;
      case bonds::PairBond::MoRe:
        return -2.096;
      case bonds::PairBond::MoNi:
        return -1.8045;
      case bonds::PairBond::MoMn:
        return -0.446;
      case bonds::PairBond::MoSi:
        return -0.683;
      case bonds::PairBond::VRe:
        return -0.920318;
      case bonds::PairBond::VNi:
        return -0.708818;
      case bonds::PairBond::VMn:
        return -0.038;
      case bonds::PairBond::VSi:
        return -0.344;
      case bonds::PairBond::ReNi:
        return -1.9095;
      case bonds::PairBond::ReMn:
        return -0.365;
      case bonds::PairBond::ReSi:
        return -0.611;
      case bonds::PairBond::MnNi:
        return -0.366;
      case bonds::PairBond::MnSi:
        return -0.488;
      case bonds::PairBond::NiSi:
        return -0.723;
      default:
        // 处理默认情况
        return 0.0;
    }
}

__device__ double _2nnMapToSwitch(bonds::PairBond::bond_type bondType){
    switch (bondType) {
      case bonds::PairBond::MoMo:
        return -0.9955;
      case bonds::PairBond::VV:
        return -0.245;
      case bonds::PairBond::ReRe:
        return -1.1005;
      case bonds::PairBond::NiNi:
        return -0.809;
      case bonds::PairBond::MnMn:
        return -0.611;
      case bonds::PairBond::SiSi:
        return -0.389;
      case bonds::PairBond::VMo:
        return -0.377659;
      case bonds::PairBond::MoRe:
        return -1.048;
      case bonds::PairBond::MoNi:
        return -0.90225;
      case bonds::PairBond::MoMn:
        return -0.631;
      case bonds::PairBond::MoSi:
        return -0.469;
      case bonds::PairBond::VRe:
        return -0.460159;
      case bonds::PairBond::VNi:
        return -0.354409;
      case bonds::PairBond::VSi:
        return -0.248;
      case bonds::PairBond::VMn:
        return -0.203;
      case bonds::PairBond::ReNi:
        return -0.95475;
      case bonds::PairBond::ReMn:
        return -0.621;
      case bonds::PairBond::ReSi:
        return -0.566;
      case bonds::PairBond::MnNi:
        return -0.496;
      case bonds::PairBond::MnSi:
        return -0.316;
      case bonds::PairBond::NiSi:
        return -0.521;
      default:
        // 处理默认情况
        return 0.0;
    }
}
__device__ int hash_set_contains_debug(GPUHashSet* table, long int key) {
  table = g_MoMo_Hash;
  if(table == nullptr)
            printf("momo is null1");
  printf("table addr:%p,key:%ld",table,key);
  if (table == nullptr) {
      printf("Error: table is nullptr\n");
      return 0;
  }
  if (table->keys == nullptr) {
      printf("Error: table->keys is nullptr\n");
      return 0;
  }
  if (table->capacity <= 0) {
      printf("Error: Invalid capacity (%d)\n", table->capacity);
      return 0;
  }

  int index = key % table->capacity;
  if (index < 0 || index >= table->capacity) {
      printf("Error: Computed index out of range (%d)\n", index);
      return 0;
  }

  printf("Searching for key: %d, start index: %d, capacity: %d\n", key, index, table->capacity);

  const int original_index = index;
  int attempts = 0;

  do {
      int current = table->keys[index];

      printf("Checking index: %d, key at index: %d\n", index, current);

      if (current == key) {
          printf("Key %d found at index %d\n", key, index);
          return 1;
      }
      if (current == table->empty_flag) {
          printf("Empty slot encountered at index %d, terminating search\n", index);
          break;  // 遇到空槽提前终止
      }

      index = (index + 1) % table->capacity;
      attempts++;

      if (attempts > table->capacity) {
          printf("Error: Infinite loop detected, breaking out\n");
          break;
      }

  } while (index != original_index);  // 完整遍历整个表

  printf("Key %d not found\n", key);
  return 0;
}


// 原子交换操作 - 支持long类型

// __device__ int hash_set_remove_atom_op(GPUHashSet* table, long key) {
//     unsigned long long index = key % table->capacity;
//     unsigned long long original_index = index;

//     volatile long* keys = (volatile long*)table->keys;
//     do {
//         long current = keys[index];
        
//         if (current == table->empty_flag) {
//             return 0;
//         }
        
//         if (current == key) {
//             long old = atomicCASLong((long*)&table->keys[index], key, table->tombstone_flag);
//             if (old == key) return 1;
//         }
        
//         index = (index + 1) % table->capacity;
//     } while (index != original_index);
    
//     return 0;
// }

// __device__ int hash_set_insert_atom_op(GPUHashSet* set, long int key) {
//   unsigned long long pos = key % set->capacity;  // 保持修正后的哈希计算
//   unsigned long long start_pos = pos;

//   volatile long* keys = (volatile long*)set->keys;
//   do {
//       long current = keys[pos];
      
//       if (current == key) {
//           return 0; // 已存在相同key
//       }
      
//       if (current == set->empty_flag || current == set->tombstone_flag) {
//           long old = atomicCASLong((long*)&set->keys[pos], current, key);
//           if (old == current) return 1;
//           if (old == key) return 0;
//       }
      
//       pos = (pos + 1) % set->capacity;
//   } while (pos != start_pos);
  
//   return 0; // 哈希表已满
// }

__global__ void calcExchangeAtoms(dev_atom *atoms, const _type_lattice_count count){

    const _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;//用这个来标志dev_atom中的原子数
    if(atoms == nullptr) return;
    if(tid < count){
      if(atoms[tid].exchange == -1){
        atoms[tid].exchange = 0;
        return;
      }
      // atoms[tid].exchange = 1;
      for(int i = 0; i < LatticesList::MAX_1NN; i++){
        if (atoms[tid].atom1nn_type[i] == LatticeTypes::MoMo || atoms[tid].atom1nn_type[i] == LatticeTypes::MoRe) atoms[tid].numSIA++;
      }
      if(atoms[tid].numSIA >= 2) {
        // printf("catched by more");
          return;
      }

      if(atoms[tid].atom_type == LatticeTypes::MoRe){
        for(int i = 0; i< LatticesList::MAX_1NN;i++){
          if(atoms[tid].atom1nn_type[i] == LatticeTypes::Re) atoms[tid].numRe++;
        }
        if (atoms[tid].numRe >= 5) {
          // printf("catched by re");
          return;
        }
      }

      _type_lattice_count randomLattice_id = 0;
      LatticeTypes::lat_type randomLattice_type = LatticeTypes::V;

      // randomLattice_id = atoms[tid].atom12nn[atoms[tid].random_num1];
      // randomLattice_type = atoms[tid].atom12nn_type[atoms[tid].random_num1];
      // if(atoms[tid].random_num1 < LatticesList::MAX_1NN){
      //   randomLattice_id = atoms[tid].atom1nn[atoms[tid].random_num1];
      //   randomLattice_type = atoms[tid].atom1nn_type[atoms[tid].random_num1];
      // }
      // else{
      //   randomLattice_id = atoms[tid].atom2nn[atoms[tid].random_num1 -  LatticesList::MAX_1NN ];
      //   randomLattice_type = atoms[tid].atom2nn_type[atoms[tid].random_num1 -  LatticesList::MAX_1NN];
      // }
      // printf("tid: %d, atomtype: %d ,randomtype: %d\n",tid,atoms[tid].atom_type,randomLattice_type);
      //获取随机数位置的粒子的信息
      // kiwi::logs::v(" ","atom type is {},random type is:{}\n",atoms[tid].atom_type,randomLattice_type);

      if (atoms[tid].exchange_type == LatticeTypes::Mo || atoms[tid].exchange_type == LatticeTypes::Re){
        if(atoms[tid].atom_type == LatticeTypes::MoMo){
          if(atoms[tid].exchange_type == LatticeTypes::Mo){   
            atoms[tid].exchange = 1;
          }
          else if(atoms[tid].exchange_type == LatticeTypes::Re){
            atoms[tid].exchange = 1;
          }
        }else if(atoms[tid].atom_type == LatticeTypes::MoRe){
          if(atoms[tid].numRe == 0){
            if(atoms[tid].exchange_type == LatticeTypes::Mo){
              atoms[tid].exchange = 1;
            }
            else{
              atoms[tid].exchange = 1;
            }
          }
          else if(atoms[tid].numRe > 0 && atoms[tid].numRe < LatticesList::MAX_1NN){
            if(atoms[tid].ranmov > (static_cast<double>(atoms[tid].numRe) / LatticesList::MAX_1NN)) {
                if (atoms[tid].exchange_type == LatticeTypes::Mo) { 
                  atoms[tid].exchange = 1;
                }
                else if(atoms[tid].exchange_type == LatticeTypes::Re){
                  atoms[tid].exchange = 1;
                }
            }
          }else return;
        }
      }
      // printf("tid: %d, atoms[tid].exchange:%d, id is %d\n",tid,atoms[tid].exchange,atoms[tid].exchange_id);
    }


}
__global__ void calcRatesGPU(dev_Vacancy *deviceVacancyInfo, dev_nnLattice *deviceNeighbourInfo,
                             double* d_sum_rates, double attempt_freq, double temperature, _type_lattice_count vac_total) {
  // 获取线程的全局ID
  _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 洗牌函数、寄存器和共享内存优化原子加
  if (tid == 0) *d_sum_rates = 0.0;
  __shared__ double tmp_sh;
  if (threadIdx.x == 0) tmp_sh = 0.0;
  __syncthreads();

  // 判断当前tid是否可用
  if(tid < vac_total) {
    double totalRatesOfCurVac = 0.0;
    
    deviceVacancyInfo[tid].rates[0] = 0;
    deviceVacancyInfo[tid].rates[1] = 0;
    deviceVacancyInfo[tid].rates[2] = 0;
    deviceVacancyInfo[tid].rates[3] = 0;
    deviceVacancyInfo[tid].rates[4] = 0;
    deviceVacancyInfo[tid].rates[5] = 0;
    deviceVacancyInfo[tid].rates[6] = 0;
    deviceVacancyInfo[tid].rates[7] = 0;

    double vac_energy = 0;
    int first_ir = 1;
    unsigned char atom_nei_status = 0;
    for (int b = 0; b < 8; b++) {
      if (deviceNeighbourInfo[tid * NN_TOTAL + b].type._type <= LatticeTypes::Si && deviceNeighbourInfo[tid * NN_TOTAL + b].type._type != LatticeTypes::V) { // can trans
        atom_nei_status |= 1 << b;
      }
    }

    for (unsigned char b = 0; b < 8; b++) {
      if ((atom_nei_status >> b) & 1) { // the neighbour lattice is available
        double active_energy;
        switch (deviceNeighbourInfo[tid * NN_TOTAL + b].type._type) {
          case LatticeTypes::Mo:
            active_energy = 0.65;
            break;
          case LatticeTypes::Re:
            active_energy = 0.54;
            break;
          case LatticeTypes::Mn:
            active_energy = 1.03;
            break;
          case LatticeTypes::Ni:
            active_energy = 0.68;
            break;
          case LatticeTypes::Si:
            active_energy = 0.42;
            break;
          default:
            active_energy = 0;
            break;
        }
        // calculate system energy before transition.
        // const LatticeTypes::lat_type ghost_atom = d_lat_list[tid * 8 + b].type._type;
        double e_before = 0;
        {
          // bonds energy of src lattice contributed by its 1nn/2nn neighbour lattice.
          double e_src, e_des;
          unsigned char _1nn_status = 0xFF;
          unsigned char _2nn_status = 0x2F;
          double energy = 0.0;
          if(first_ir == 1) {
            // Traver all 1nn neighbour lattices, and calculate bond energy contribution.
            for (int b2 = 0; b2 < 8; b2++) {
              if ((_1nn_status >> b2) & 0x1) {
                // we assume that src_atom_type is single atom or vacancy.
                if (deviceNeighbourInfo[tid * NN_TOTAL + b2].type._type <= LatticeTypes::Si) {
                  // it is vacancy or single atom.
                  LatticeTypes::lat_type temp_1nn = deviceNeighbourInfo[tid * NN_TOTAL + b2].type._type;
                  LatticeTypes::lat_type temp_src = LatticeTypes::V;
                  if (temp_1nn > temp_src) {
                    LatticeTypes::lat_type temp = temp_1nn;
                    temp_1nn = temp_src;
                    temp_src = temp;
                  }
                  // PairBond::bond_type bond = PairBond::makeBond(_1nn_neighbour[b]->type, src_atom_type);
                  const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_1nn << 8) | temp_src);
                  vac_energy += _1nnMapToSwitch(bond);
                }
              }
            }
        
            // Traver all 2nn neighbour lattices, and calculate bond energy contribution.
            for (int b2 = 0; b2 < 6; b2++) {
              if ((_2nn_status >> b2) & 0x1) {
                // we assume that src_atom_type is single atom or vacancy.

                if (deviceNeighbourInfo[tid * NN_TOTAL + 8 + b2].type._type <= LatticeTypes::Si) {
                  // it is vacancy or single atom.
                  LatticeTypes::lat_type temp_2nn = deviceNeighbourInfo[tid * NN_TOTAL + 8 + b2].type._type;
                  LatticeTypes::lat_type temp_src = LatticeTypes::V;
                  if (temp_2nn > temp_src) {
                    LatticeTypes::lat_type temp = temp_2nn;
                    temp_2nn = temp_src;
                    temp_src = temp;
                  }
                  // PairBond::bond_type bond = PairBond::makeBond(_2nn_neighbour[b3]->type, src_atom_type);
                  const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_2nn << 8) | temp_src);
                  vac_energy += _2nnMapToSwitch(bond);
                }
              }
            }
            first_ir = 0;
          }
          e_src = vac_energy;
//_______________________________________
          energy = 0.0;

          for (int b2 = 0; b2 < 8; b2++) {
            if ((_1nn_status >> b2) & 0x1) {
              // we assume that src_atom_type is single atom or vacancy.
              if (deviceNeighbourInfo[tid * NN_TOTAL + 14 + b * 8 + b2].type._type <= LatticeTypes::Si) {
                // it is vacancy or single atom.
                LatticeTypes::lat_type temp_1nn = deviceNeighbourInfo[tid * NN_TOTAL + 14 + b * 8 + b2].type._type;
                LatticeTypes::lat_type temp_src = deviceNeighbourInfo[tid * NN_TOTAL + b].type._type;
                if (temp_1nn > temp_src) {
                  LatticeTypes::lat_type temp = temp_1nn;
                  temp_1nn = temp_src;
                  temp_src = temp;
                }
                // PairBond::bond_type bond = PairBond::makeBond(_1nn_neighbour[b]->type, src_atom_type);
                const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_1nn << 8) | temp_src);
                energy += _1nnMapToSwitch(bond);
              }
            }
          }

          for (int b2 = 0; b2 < 6; b2++) {
            if ((_2nn_status >> b2) & 0x1) {
              // we assume that src_atom_type is single atom or vacancy.
              if (deviceNeighbourInfo[tid * NN_TOTAL + 78 + b * 6 + b2].type._type <= LatticeTypes::Si) {
                // it is vacancy or single atom.
                LatticeTypes::lat_type temp_2nn = deviceNeighbourInfo[tid * NN_TOTAL + 78 + b * 6 + b2].type._type;
                LatticeTypes::lat_type temp_src = deviceNeighbourInfo[tid * NN_TOTAL + b].type._type;
                if (temp_2nn > temp_src) {
                  LatticeTypes::lat_type temp = temp_2nn;
                  temp_2nn = temp_src;
                  temp_src = temp;
                }
                // PairBond::bond_type bond = PairBond::makeBond(_2nn_neighbour[b3]->type, src_atom_type);
                const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_2nn << 8) | temp_src);
                energy += _2nnMapToSwitch(bond);
              }
            }
          }
          // if(is_first) *energy_temp2 = energy;
          // is_first = 0;
          e_des = energy;

          // double e_des = bonds::BondsCounter::count(&lattice_list, target_id, target_lattice.type);
          e_before = e_src + e_des;
          // energy_temp2[b] = energy;
        }

// ______________________________________________________________________________________________________________________________________



        // calculate system energy after transition.

        double e_after = 0;
        {
          // bonds energy of src lattice contributed by its 1nn/2nn neighbour lattice.
          double e_src, e_des;
          unsigned char _1nn_status = 0xFF;
              
          double energy = 0.0;

          for (int b2 = 0; b2 < 8; b2++) {
            if ((_1nn_status >> b2) & 0x1) {
              // we assume that src_atom_type is single atom or vacancy.
              LatticeTypes::lat_type temp_type = deviceNeighbourInfo[tid * NN_TOTAL + b2].type._type;
              if (b2 == b){
                temp_type = LatticeTypes::V;
              }
              if (temp_type <= LatticeTypes::Si) {
                // it is vacancy or single atom.
                LatticeTypes::lat_type temp_1nn = temp_type; 
                LatticeTypes::lat_type temp_src = deviceNeighbourInfo[tid * NN_TOTAL + b].type._type;
                if (temp_1nn > temp_src) {
                  LatticeTypes::lat_type temp = temp_1nn;
                  temp_1nn = temp_src;
                  temp_src = temp;
                }
                // PairBond::bond_type bond = PairBond::makeBond(_1nn_neighbour[b]->type, src_atom_type);
                bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_1nn << 8) | temp_src);
                energy += _1nnMapToSwitch(bond);
              }
            }
          }

          // Traver all 2nn neighbour lattices, and calculate bond energy contribution.
          unsigned char _2nn_status = 0x2F;
          
          // lat_list->get2nn(source_id, _2nn_neighbour);
          for (int b2 = 0; b2 < 6; b2++) {
            if ((_2nn_status >> b2) & 0x1) {
              // we assume that src_atom_type is single atom or vacancy.
              if (deviceNeighbourInfo[tid * NN_TOTAL + 8 + b2].type._type <= LatticeTypes::Si) {
                // it is vacancy or single atom.
                LatticeTypes::lat_type temp_2nn = deviceNeighbourInfo[tid * NN_TOTAL + 8 + b2].type._type;
                LatticeTypes::lat_type temp_src = deviceNeighbourInfo[tid * NN_TOTAL + b].type._type;
                if (temp_2nn > temp_src) {
                  LatticeTypes::lat_type temp = temp_2nn;
                  temp_2nn = temp_src;
                  temp_src = temp;
                }
                // PairBond::bond_type bond = PairBond::makeBond(_2nn_neighbour[b3]->type, src_atom_type);
                const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_2nn << 8) | temp_src);
                energy += _2nnMapToSwitch(bond);
              }
            }
          }
          e_src = energy;
//_______________________________________
          energy = 0.0;
          for (int b2 = 0; b2 < 8; b2++) {
            if ((_1nn_status >> b2) & 0x1) {
              // we assume that src_atom_type is single atom or vacancy.
              LatticeTypes::lat_type temp_type = deviceNeighbourInfo[tid * NN_TOTAL + 14 + b * 8 + b2].type._type;
              if (b2 + b == 7) {
                temp_type = deviceNeighbourInfo[tid * NN_TOTAL + b].type._type;
              }
              // if (d_target_lattice_1neighbour[tid * 64 + b * 8 + b2].id == d_lattices[tid].id){
              //   temp_type = d_lat_list[tid * 8 + b].type._type;
              // }
              if (temp_type <= LatticeTypes::Si) {
                // it is vacancy or single atom.
                LatticeTypes::lat_type temp_1nn = temp_type;
                LatticeTypes::lat_type temp_src = LatticeTypes::V;
                if (temp_1nn > temp_src) {
                  LatticeTypes::lat_type temp = temp_1nn;
                  temp_1nn = temp_src;
                  temp_src = temp;
                }
                // PairBond::bond_type bond = PairBond::makeBond(_1nn_neighbour[b]->type, src_atom_type);
                const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_1nn << 8) | temp_src);
                energy += _1nnMapToSwitch(bond);
              }
            }
          }
          for (int b2 = 0; b2 < 6; b2++) {
            if ((_2nn_status >> b2) & 0x1) {
              // we assume that src_atom_type is single atom or vacancy.
              if (deviceNeighbourInfo[tid * NN_TOTAL + 78 + b * 6 + b2].type._type <= LatticeTypes::Si) {
                // it is vacancy or single atom.
                LatticeTypes::lat_type temp_2nn = deviceNeighbourInfo[tid * NN_TOTAL + 78 + b * 6 + b2].type._type;;
                LatticeTypes::lat_type temp_src = LatticeTypes::V;
                if (temp_2nn > temp_src) {
                  LatticeTypes::lat_type temp = temp_2nn;
                  temp_2nn = temp_src;
                  temp_src = temp;
                }
                // PairBond::bond_type bond = PairBond::makeBond(_2nn_neighbour[b3]->type, src_atom_type);
                const bonds::PairBond::bond_type bond = static_cast<bonds::PairBond::bond_type>((temp_2nn << 8) | temp_src);
                energy += _2nnMapToSwitch(bond);
              }
            }
          }
          e_des = energy;
          e_after = e_src + e_des;
        }
        active_energy += (e_after - e_before) / 2;
        double rate = attempt_freq * __expf(-active_energy / (8.617330350E-5 * temperature));
        deviceVacancyInfo[tid].rates[b] = rate;
        totalRatesOfCurVac += rate;
      }
    }
    int delta = 1;
    for (int j = 0; j < 6; j++) {
      totalRatesOfCurVac += __shfl_down(totalRatesOfCurVac, delta, 64);
      delta += delta;
    }
    if (threadIdx.x % 64 == 0) atomicAdd(&tmp_sh, totalRatesOfCurVac);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(d_sum_rates, tmp_sh);
  }
}

__global__ void selectEventGPU(dev_Vacancy *deviceVacancyInfo, dev_nnLattice *deviceNeighbourInfo, 
                               dev_event *d_event, _type_lattice_count vac_total, double excepted_rate) {

  // 获取线程的全局ID
  _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid == 0) {
    double rate_accumulator = 0;
    int tag = 1;
    for(_type_lattice_count i = 0; i < vac_total; i++) {
      if(tag == 1) {
        for (int rate_index = 0; rate_index < 8; rate_index++) {
          rate_accumulator += deviceVacancyInfo[i].rates[rate_index];
          if(rate_accumulator > excepted_rate) {
            if(tag == 1) {
              d_event->from_id = deviceVacancyInfo[i].id;
              d_event->to_id = deviceNeighbourInfo[i * NN_TOTAL + rate_index].id;
              d_event->to_type._type = deviceNeighbourInfo[i * NN_TOTAL + rate_index].type._type;
              tag = 0;
              break;
            }
          }
        } 
      } else break;
    }
  }
}

__global__ void apply_selected_event_kernel(const dev_event* ev, int sector_id) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  dev_event e = ev[0];
  long int from_id = e.from_id;
  long int to_id = e.to_id;
  LatticeTypes::lat_type to_type = static_cast<LatticeTypes::lat_type>(e.to_type._type);

  hash_set_remove_atom_op(g_V_Hash, from_id);
  hash_set_insert_atom_op(g_V_Hash, to_id);

  switch (to_type) {
    case LatticeTypes::MoRe:
      hash_set_remove_atom_op(g_MoRe_Hash, to_id);
      hash_set_insert_atom_op(g_MoRe_Hash, from_id);
      break;
    case LatticeTypes::MoMo:
      hash_set_remove_atom_op(g_MoMo_Hash, to_id);
      hash_set_insert_atom_op(g_MoMo_Hash, from_id);
      break;
    case LatticeTypes::Re:
      hash_set_remove_atom_op(g_Re_Hash, to_id);
      hash_set_insert_atom_op(g_Re_Hash, from_id);
      break;
    case LatticeTypes::V:
      break;
    case LatticeTypes::Mo:
      break;
    default:
      break;
  }

  if (isGhostLat(to_id)) {
    _type_lattice_id x = to_id % g_meta->size_x;
    _type_lattice_id y = (to_id / g_meta->size_x) % g_meta->size_y;
    _type_lattice_id z = to_id / (g_meta->size_x * g_meta->size_y);
    int base = sector_id * 7;
    for (int k = 0; k < 7; k++) {
      const auto* r = &g_comm_ghost_regions[base + k];
      if (r->x_low <= x && x < r->x_high && r->y_low <= y && y < r->y_high && r->z_low <= z && z < r->z_high) {
        GPUHashSet* b = commBucketPtr(k);
        if (b) hash_set_insert_atom_op(b, to_id);
        break;
      }
    }
  }

  if (isSurfaceLat(from_id)) {
    _type_lattice_id x = from_id % g_meta->size_x;
    _type_lattice_id y = (from_id / g_meta->size_x) % g_meta->size_y;
    _type_lattice_id z = from_id / (g_meta->size_x * g_meta->size_y);
    if (g_comm_surface_offsets && g_comm_surface_boxes && g_comm_surface_buckets_ptr) {
      for (int b = 0; b < 24; b++) {
        int start = g_comm_surface_offsets[b];
        int end = g_comm_surface_offsets[b + 1];
        for (int j = start; j < end; j++) {
          CommBox box = g_comm_surface_boxes[j];
          if (box.x_low <= x && x < box.x_high && box.y_low <= y && y < box.y_high && box.z_low <= z && z < box.z_high) {
            GPUHashSet* set = g_comm_surface_buckets_ptr[b];
            if (set) hash_set_insert_atom_op(set, from_id);
            break;
          }
        }
      }
    }
  }

  if (isSurfaceLat(to_id)) {
    _type_lattice_id x = to_id % g_meta->size_x;
    _type_lattice_id y = (to_id / g_meta->size_x) % g_meta->size_y;
    _type_lattice_id z = to_id / (g_meta->size_x * g_meta->size_y);
    if (g_comm_surface_offsets && g_comm_surface_boxes && g_comm_surface_buckets_ptr) {
      for (int b = 0; b < 24; b++) {
        int start = g_comm_surface_offsets[b];
        int end = g_comm_surface_offsets[b + 1];
        for (int j = start; j < end; j++) {
          CommBox box = g_comm_surface_boxes[j];
          if (box.x_low <= x && x < box.x_high && box.y_low <= y && y < box.y_high && box.z_low <= z && z < box.z_high) {
            GPUHashSet* set = g_comm_surface_buckets_ptr[b];
            if (set) hash_set_insert_atom_op(set, to_id);
            break;
          }
        }
      }
    }
  }
}

void selectAndPerformEventGPU(double excepted_rand, int rank, _type_lattice_count step, int sect,
                              std::array<std::unordered_set<_type_lattice_id>, 7>& exchange_ghost, const unsigned int sector_id,
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_x, 
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_y,
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_z,
                              std::array<comm::Region<comm::_type_lattice_size>, 56>& ghost_region){
  using namespace gpuSimulate;
  commSetGhostRegions(&ghost_region[0], 56);

  blockSize = 1;
  numBlocks = 1;
  hipLaunchKernelGGL(selectEventGPU, numBlocks, blockSize, 0, 0, 
                    deviceVacancyInfo,
                    deviceNeighbourInfo,
                    d_event,
                    currentVacCount,
                     *sum_rates * excepted_rand);

  CHECK_FOR_ERROR();
  if (gpu_comm_enabled()) {
    hipLaunchKernelGGL(apply_selected_event_kernel, 1, 1, 0, 0, d_event, static_cast<int>(sector_id));
    CHECK_FOR_ERROR();
    return;
  }

  HANDLE_HIP(hipMemcpy(h_event, d_event, sizeof(dev_event), hipMemcpyDeviceToHost));
  to_x = h_event->to_id % lattice_list->meta.size_x;
  to_y = (h_event->to_id / lattice_list->meta.size_x) % lattice_list->meta.size_y;
  to_z = h_event->to_id / (lattice_list->meta.size_x * lattice_list->meta.size_y);

  exchange_lattices(h_event);
  if(lattice_list->meta.isGhostLat(h_event->to_id)) {
    for(int i = 0; i < 7; i++) {
      if(ghost_region[sector_id * 7 + i].x_low <= to_x && to_x < ghost_region[sector_id * 7 + i].x_high && 
         ghost_region[sector_id * 7 + i].y_low <= to_y && to_y < ghost_region[sector_id * 7 + i].y_high && 
         ghost_region[sector_id * 7 + i].z_low <= to_z && to_z < ghost_region[sector_id * 7 + i].z_high) {
          if(!exchange_ghost[i].count(h_event->to_id)) exchange_ghost[i].emplace(h_event->to_id);
          break;
      }
    }
  }
  if (lattice_list->meta.isSurfaceLat(h_event->from_id)) add_surface(h_event->from_id, exchange_surface_x, exchange_surface_y, exchange_surface_z);
  if (lattice_list->meta.isSurfaceLat(h_event->to_id)) add_surface(h_event->to_id, exchange_surface_x, exchange_surface_y, exchange_surface_z);
}
__global__ void setGlobalPointer(dev_meta *meta,
                            GPUHashSet *MoRe_Hash,GPUHashSet *MoMo_Hash,
                            GPUHashSet *Re_Hash,GPUHashSet *V_Hash,GPUHashSet *Busy_Set_Hash,
                            GPUHashSet *surface_Hash,GPUHashSet *ghost_Hash,
                            long int *Pair_Atoms,
                            unsigned long long int* d_bitmap_lock
                            ){
  if (threadIdx.x == 0 && blockIdx.x == 0) {
  g_meta = meta;
  g_MoRe_Hash = MoRe_Hash;
  g_MoMo_Hash = MoMo_Hash;
  // printf("MoMo:%p\n",MoMo_Hash);
  // printf("g_MoMo:%p\n",g_MoMo_Hash);
  g_Re_Hash = Re_Hash;
  g_V_Hash = V_Hash;
  g_Busy_Set = Busy_Set_Hash;
  g_surface_Hash = surface_Hash;
  g_ghost_Hash = ghost_Hash;
  g_Pair_Atoms = Pair_Atoms;
  g_Global_Bitmap_Lock = d_bitmap_lock;
  
    // printHashTableKernel(g_MoMo_Hash,200,false);
    // printDevMetaKernel(g_meta);

   
  }
}

__global__ void setCommGhostPointers(GPUHashSet *b0, GPUHashSet *b1, GPUHashSet *b2, GPUHashSet *b3, GPUHashSet *b4, GPUHashSet *b5, GPUHashSet *b6,
                                     comm::Region<comm::_type_lattice_size> *ghost_regions) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    g_comm_ghost_bucket0 = b0;
    g_comm_ghost_bucket1 = b1;
    g_comm_ghost_bucket2 = b2;
    g_comm_ghost_bucket3 = b3;
    g_comm_ghost_bucket4 = b4;
    g_comm_ghost_bucket5 = b5;
    g_comm_ghost_bucket6 = b6;
    g_comm_ghost_regions = ghost_regions;
  }
}

static __device__ __forceinline__ GPUHashSet* commBucketPtr(int idx) {
  switch (idx) {
    case 0: return g_comm_ghost_bucket0;
    case 1: return g_comm_ghost_bucket1;
    case 2: return g_comm_ghost_bucket2;
    case 3: return g_comm_ghost_bucket3;
    case 4: return g_comm_ghost_bucket4;
    case 5: return g_comm_ghost_bucket5;
    case 6: return g_comm_ghost_bucket6;
    default: return nullptr;
  }
}

__global__ void bucketizeGhostIdsKernel(const long int *ids, int n, int sector_id) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long int id = ids[i];
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  int base = sector_id * 7;
  for (int k = 0; k < 7; k++) {
    const auto* r = &g_comm_ghost_regions[base + k];
    if (r->x_low <= x && x < r->x_high && r->y_low <= y && y < r->y_high && r->z_low <= z && z < r->z_high) {
      GPUHashSet* b = commBucketPtr(k);
      if (b) hash_set_insert_atom_op(b, id);
      break;
    }
  }
}

__global__ void insertGhostIdKernel(long int id, int sector_id) {
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  int base = sector_id * 7;
  for (int k = 0; k < 7; k++) {
    const auto* r = &g_comm_ghost_regions[base + k];
    if (r->x_low <= x && x < r->x_high && r->y_low <= y && y < r->y_high && r->z_low <= z && z < r->z_high) {
      GPUHashSet* b = commBucketPtr(k);
      if (b) hash_set_insert_atom_op(b, id);
      break;
    }
  }
}

__global__ void compactHashSetKernel(GPUHashSet *set, long int *out, int *out_count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!set) return;
  if (i >= set->capacity) return;
  long int key = set->keys[i];
  if (key == set->empty_flag || key == set->tombstone_flag) return;
  int pos = atomicAdd(out_count, 1);
  out[pos] = key;
}

__global__ void compactHashSetBoundedKernel(GPUHashSet *set, long int *out, int *out_count, int max_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!set || !out || !out_count) return;
  if (i >= set->capacity) return;
  long int key = set->keys[i];
  if (key == set->empty_flag || key == set->tombstone_flag) return;
  int pos = atomicAdd(out_count, 1);
  if (pos < 0 || pos >= max_out) return;
  out[pos] = key;
}

void commSetGhostRegions(const comm::Region<comm::_type_lattice_size> *ghost_region, int n) {
  if (!ghost_region) return;
  if (n <= 0) return;
  if (gpuSimulate::d_comm_ghost_regions == nullptr) {
    HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&gpuSimulate::d_comm_ghost_regions), static_cast<size_t>(n) * sizeof(comm::Region<comm::_type_lattice_size>)));
  }
  HANDLE_HIP(hipMemcpy(gpuSimulate::d_comm_ghost_regions, ghost_region, static_cast<size_t>(n) * sizeof(comm::Region<comm::_type_lattice_size>), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(setCommGhostPointers, 1, 1, 0, 0,
                     gpuSimulate::comm_ghost_bucket0 ? gpuSimulate::comm_ghost_bucket0->device_ptr() : nullptr,
                     gpuSimulate::comm_ghost_bucket1 ? gpuSimulate::comm_ghost_bucket1->device_ptr() : nullptr,
                     gpuSimulate::comm_ghost_bucket2 ? gpuSimulate::comm_ghost_bucket2->device_ptr() : nullptr,
                     gpuSimulate::comm_ghost_bucket3 ? gpuSimulate::comm_ghost_bucket3->device_ptr() : nullptr,
                     gpuSimulate::comm_ghost_bucket4 ? gpuSimulate::comm_ghost_bucket4->device_ptr() : nullptr,
                     gpuSimulate::comm_ghost_bucket5 ? gpuSimulate::comm_ghost_bucket5->device_ptr() : nullptr,
                     gpuSimulate::comm_ghost_bucket6 ? gpuSimulate::comm_ghost_bucket6->device_ptr() : nullptr,
                     gpuSimulate::d_comm_ghost_regions);
  hipDeviceSynchronize();
}

void commInitGhostBuckets(int capacity) {
  if (capacity <= 0) return;
  gpuSimulate::comm_ghost_capacity = capacity * 2;
  if (!gpuSimulate::comm_ghost_bucket0) gpuSimulate::comm_ghost_bucket0 = new HIPHashSet(capacity);
  if (!gpuSimulate::comm_ghost_bucket1) gpuSimulate::comm_ghost_bucket1 = new HIPHashSet(capacity);
  if (!gpuSimulate::comm_ghost_bucket2) gpuSimulate::comm_ghost_bucket2 = new HIPHashSet(capacity);
  if (!gpuSimulate::comm_ghost_bucket3) gpuSimulate::comm_ghost_bucket3 = new HIPHashSet(capacity);
  if (!gpuSimulate::comm_ghost_bucket4) gpuSimulate::comm_ghost_bucket4 = new HIPHashSet(capacity);
  if (!gpuSimulate::comm_ghost_bucket5) gpuSimulate::comm_ghost_bucket5 = new HIPHashSet(capacity);
  if (!gpuSimulate::comm_ghost_bucket6) gpuSimulate::comm_ghost_bucket6 = new HIPHashSet(capacity);

  gpuSimulate::comm_ghost_capacity = gpuSimulate::comm_ghost_bucket0->get_capacity();
  hipLaunchKernelGGL(setCommGhostPointers, 1, 1, 0, 0,
                     gpuSimulate::comm_ghost_bucket0->device_ptr(),
                     gpuSimulate::comm_ghost_bucket1->device_ptr(),
                     gpuSimulate::comm_ghost_bucket2->device_ptr(),
                     gpuSimulate::comm_ghost_bucket3->device_ptr(),
                     gpuSimulate::comm_ghost_bucket4->device_ptr(),
                     gpuSimulate::comm_ghost_bucket5->device_ptr(),
                     gpuSimulate::comm_ghost_bucket6->device_ptr(),
                     gpuSimulate::d_comm_ghost_regions);
  hipDeviceSynchronize();
}

void commClearGhostBuckets() {
  if (gpuSimulate::comm_ghost_bucket0) gpuSimulate::comm_ghost_bucket0->clear();
  if (gpuSimulate::comm_ghost_bucket1) gpuSimulate::comm_ghost_bucket1->clear();
  if (gpuSimulate::comm_ghost_bucket2) gpuSimulate::comm_ghost_bucket2->clear();
  if (gpuSimulate::comm_ghost_bucket3) gpuSimulate::comm_ghost_bucket3->clear();
  if (gpuSimulate::comm_ghost_bucket4) gpuSimulate::comm_ghost_bucket4->clear();
  if (gpuSimulate::comm_ghost_bucket5) gpuSimulate::comm_ghost_bucket5->clear();
  if (gpuSimulate::comm_ghost_bucket6) gpuSimulate::comm_ghost_bucket6->clear();
}

void commBuildGhostBucketsFromList(int sector_id, const long int *d_ids, int n) {
  if (!d_ids) return;
  if (n <= 0) return;
  int blockSize = 256;
  int blockNum = (n + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(bucketizeGhostIdsKernel, blockNum, blockSize, 0, 0, d_ids, n, sector_id);
  hipDeviceSynchronize();
}

void commInsertGhostId(int sector_id, long int id) {
  hipLaunchKernelGGL(insertGhostIdKernel, 1, 1, 0, 0, id, sector_id);
}

void commCompactGhostBucket(int bucket, long int *d_out, int *d_count) {
  if (!d_out || !d_count) return;
  HANDLE_HIP(hipMemset(d_count, 0, sizeof(int)));
  HIPHashSet *b = nullptr;
  switch (bucket) {
    case 0: b = gpuSimulate::comm_ghost_bucket0; break;
    case 1: b = gpuSimulate::comm_ghost_bucket1; break;
    case 2: b = gpuSimulate::comm_ghost_bucket2; break;
    case 3: b = gpuSimulate::comm_ghost_bucket3; break;
    case 4: b = gpuSimulate::comm_ghost_bucket4; break;
    case 5: b = gpuSimulate::comm_ghost_bucket5; break;
    case 6: b = gpuSimulate::comm_ghost_bucket6; break;
    default: b = nullptr; break;
  }
  if (!b) return;
  int cap = gpuSimulate::comm_ghost_capacity;
  int blockSize = 256;
  int blockNum = (cap + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(compactHashSetKernel, blockNum, blockSize, 0, 0, b->device_ptr(), d_out, d_count);
  hipDeviceSynchronize();
}

int commGhostCapacity() {
  return gpuSimulate::comm_ghost_capacity;
}

__global__ void setCommSurfaceMeta(GPUHashSet **bucket_ptrs, CommBox *boxes, int *offsets) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    g_comm_surface_buckets_ptr = bucket_ptrs;
    g_comm_surface_boxes = boxes;
    g_comm_surface_offsets = offsets;
  }
}

__global__ void bucketizeSurfaceIdsKernel(const long int *ids, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long int id = ids[i];
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  if (!g_comm_surface_offsets || !g_comm_surface_boxes || !g_comm_surface_buckets_ptr) return;
  for (int b = 0; b < 24; b++) {
    int start = g_comm_surface_offsets[b];
    int end = g_comm_surface_offsets[b + 1];
    for (int j = start; j < end; j++) {
      CommBox box = g_comm_surface_boxes[j];
      if (box.x_low <= x && x < box.x_high && box.y_low <= y && y < box.y_high && box.z_low <= z && z < box.z_high) {
        GPUHashSet* set = g_comm_surface_buckets_ptr[b];
        if (set) hash_set_insert_atom_op(set, id);
        break;
      }
    }
  }
}

__global__ void insertSurfaceIdKernel(long int id) {
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  if (!g_comm_surface_offsets || !g_comm_surface_boxes || !g_comm_surface_buckets_ptr) return;
  for (int b = 0; b < 24; b++) {
    int start = g_comm_surface_offsets[b];
    int end = g_comm_surface_offsets[b + 1];
    for (int j = start; j < end; j++) {
      CommBox box = g_comm_surface_boxes[j];
      if (box.x_low <= x && x < box.x_high && box.y_low <= y && y < box.y_high && box.z_low <= z && z < box.z_high) {
        GPUHashSet* set = g_comm_surface_buckets_ptr[b];
        if (set) hash_set_insert_atom_op(set, id);
        break;
      }
    }
  }
}

void commSetSurfaceBoxes(const CommBox *boxes, int boxes_count, const int *offsets, int offsets_count) {
  if (boxes_count < 0 || offsets_count <= 0) return;
  if (boxes_count > 0) {
    if (gpuSimulate::d_comm_surface_boxes && gpuSimulate::comm_surface_boxes_cap < boxes_count) {
      HANDLE_HIP(hipFree(gpuSimulate::d_comm_surface_boxes));
      gpuSimulate::d_comm_surface_boxes = nullptr;
      gpuSimulate::comm_surface_boxes_cap = 0;
    }
    if (!gpuSimulate::d_comm_surface_boxes) {
      HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&gpuSimulate::d_comm_surface_boxes), static_cast<size_t>(boxes_count) * sizeof(CommBox)));
      gpuSimulate::comm_surface_boxes_cap = boxes_count;
    }
    HANDLE_HIP(hipMemcpy(gpuSimulate::d_comm_surface_boxes, boxes, static_cast<size_t>(boxes_count) * sizeof(CommBox), hipMemcpyHostToDevice));
  }
  if (gpuSimulate::d_comm_surface_offsets && gpuSimulate::comm_surface_offsets_cap < offsets_count) {
    HANDLE_HIP(hipFree(gpuSimulate::d_comm_surface_offsets));
    gpuSimulate::d_comm_surface_offsets = nullptr;
    gpuSimulate::comm_surface_offsets_cap = 0;
  }
  if (!gpuSimulate::d_comm_surface_offsets) {
    HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&gpuSimulate::d_comm_surface_offsets), static_cast<size_t>(offsets_count) * sizeof(int)));
    gpuSimulate::comm_surface_offsets_cap = offsets_count;
  }
  HANDLE_HIP(hipMemcpy(gpuSimulate::d_comm_surface_offsets, offsets, static_cast<size_t>(offsets_count) * sizeof(int), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(setCommSurfaceMeta, 1, 1, 0, 0, gpuSimulate::d_comm_surface_bucket_ptrs, gpuSimulate::d_comm_surface_boxes, gpuSimulate::d_comm_surface_offsets);
  hipDeviceSynchronize();
}

void commInitSurfaceBuckets(int capacity) {
  if (capacity <= 0) return;
  gpuSimulate::comm_surface_capacity = capacity * 2;
  bool flag = false;
  for (int i = 0; i < 24; i++) {
    if (!gpuSimulate::comm_surface_buckets[i]){
      gpuSimulate::comm_surface_buckets[i] = new HIPHashSet(capacity);
      if(!flag){
        gpuSimulate::comm_surface_capacity = gpuSimulate::comm_surface_buckets[i]->get_capacity();
        flag=true;
      }
    } 
  }
  

  if (!gpuSimulate::d_comm_surface_bucket_ptrs) {
    HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&gpuSimulate::d_comm_surface_bucket_ptrs), 24 * sizeof(GPUHashSet*)));
  }
  GPUHashSet* h_ptrs[24];
  for (int i = 0; i < 24; i++) h_ptrs[i] = gpuSimulate::comm_surface_buckets[i]->device_ptr();
  HANDLE_HIP(hipMemcpy(gpuSimulate::d_comm_surface_bucket_ptrs, h_ptrs, 24 * sizeof(GPUHashSet*), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(setCommSurfaceMeta, 1, 1, 0, 0, gpuSimulate::d_comm_surface_bucket_ptrs, gpuSimulate::d_comm_surface_boxes, gpuSimulate::d_comm_surface_offsets);
  hipDeviceSynchronize();
}

void commClearSurfaceBuckets() {
  for (int i = 0; i < 24; i++) {
    if (gpuSimulate::comm_surface_buckets[i]) gpuSimulate::comm_surface_buckets[i]->clear();
  }
}

void commClearSurfaceSector(int sect) {
  if (sect < 0 || sect >= 8) return;
  int base = sect * 3;
  for (int i = 0; i < 3; i++) {
    int idx = base + i;
    if (gpuSimulate::comm_surface_buckets[idx]) gpuSimulate::comm_surface_buckets[idx]->clear();
  }
}

void commBuildSurfaceBucketsFromList(const long int *d_ids, int n) {
  if (!d_ids) return;
  if (n <= 0) return;
  int blockSize = 256;
  int blockNum = (n + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(bucketizeSurfaceIdsKernel, blockNum, blockSize, 0, 0, d_ids, n);
  hipDeviceSynchronize();
}

void commInsertSurfaceId(long int id) {
  hipLaunchKernelGGL(insertSurfaceIdKernel, 1, 1, 0, 0, id);
}

void commCompactSurfaceBucket(int bucket, long int *d_out, int *d_count) {
  if (!d_out || !d_count) return;
  if (bucket < 0 || bucket >= 24) return;
  HANDLE_HIP(hipMemset(d_count, 0, sizeof(int)));
  HIPHashSet *b = gpuSimulate::comm_surface_buckets[bucket];
  if (!b) return;
  int cap = gpuSimulate::comm_surface_capacity;
  int blockSize = 256;
  int blockNum = (cap + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(compactHashSetKernel, blockNum, blockSize, 0, 0, b->device_ptr(), d_out, d_count);
  hipDeviceSynchronize();
}

int commSurfaceCapacity() {
  return gpuSimulate::comm_surface_capacity;
}

__device__ int isGhostLat(_type_lattice_id lid);
__device__ int isSurfaceLat(_type_lattice_id lid);

__global__ void forwardFromRecvPayloadKernel(const ChangeLattice_GPU *payload, int n, int sector_id) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  ChangeLattice_GPU c = payload[i];
  long int id = getId(c.x, c.y, c.z);
  if (isGhostLat(id)) {
    int base = sector_id * 7;
    for (int k = 0; k < 7; k++) {
      const auto* r = &g_comm_ghost_regions[base + k];
      if (r->x_low <= c.x && c.x < r->x_high && r->y_low <= c.y && c.y < r->y_high && r->z_low <= c.z && c.z < r->z_high) {
        GPUHashSet* b = commBucketPtr(k);
        if (b) hash_set_insert_atom_op(b, id);
        break;
      }
    }
  }
  if (isSurfaceLat(id)) {
    if (!g_comm_surface_offsets || !g_comm_surface_boxes || !g_comm_surface_buckets_ptr) return;
    for (int b = 0; b < 24; b++) {
      int start = g_comm_surface_offsets[b];
      int end = g_comm_surface_offsets[b + 1];
      for (int j = start; j < end; j++) {
        CommBox box = g_comm_surface_boxes[j];
        if (box.x_low <= c.x && c.x < box.x_high && box.y_low <= c.y && c.y < box.y_high && box.z_low <= c.z && c.z < box.z_high) {
          GPUHashSet* set = g_comm_surface_buckets_ptr[b];
          if (set) hash_set_insert_atom_op(set, id);
          break;
        }
      }
    }
  }
}

void commForwardFromRecvPayload(const ChangeLattice_GPU *d_payload, int n, int sector_id) {
  if (!d_payload) return;
  if (n <= 0) return;
  int blockSize = 256;
  int blockNum = (n + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(forwardFromRecvPayloadKernel, blockNum, blockSize, 0, 0, d_payload, n, sector_id);
}
__global__ void printpointer(){
  if(g_MoMo_Hash == nullptr)
    printf("is a null\n");
  else
    printf("not null\n");
}
__device__ LatticeTypes::lat_type getType(const long int& id){
      // printf("START GET TYPE");
      if(hash_set_contains(g_MoRe_Hash,id))       return LatticeTypes::MoRe;
      else if(hash_set_contains(g_MoMo_Hash,id))  return LatticeTypes::MoMo;
      else if(hash_set_contains(g_Re_Hash,id))    return LatticeTypes::Re;
      else if(hash_set_contains(g_V_Hash,id))     return LatticeTypes::V;
      else return LatticeTypes::Mo;

}

__device__ _type_lattice_id getId(_type_lattice_size x,_type_lattice_size y,_type_lattice_size z){
  _type_lattice_id res = x + g_meta->size_x * (y + z * g_meta->size_y);
  return (res<0 ? 0 : res);
  // return x + g_meta->size_x * (y + z * g_meta->size_y);
}
__device__ void getnnlattice(_type_lattice_id latti_id, dev_Lattice *_1nn_list_lattice){
  _1nn_list_lattice->id = latti_id;
  if(hash_set_contains(g_MoRe_Hash,latti_id))        _1nn_list_lattice->type = LatticeTypes::MoRe;
  else if(hash_set_contains(g_MoMo_Hash,latti_id))   _1nn_list_lattice->type = LatticeTypes::MoMo;
  else if(hash_set_contains(g_Re_Hash,latti_id))     _1nn_list_lattice->type = LatticeTypes::Re;
  else if(hash_set_contains(g_V_Hash,latti_id))      _1nn_list_lattice->type = LatticeTypes::V;
  else _1nn_list_lattice->type = LatticeTypes::Mo;

}
__device__ bool single_chk_type(long int id, LatticeTypes::lat_type dcl_type) {
    GPUHashSet* target_hash = nullptr;

    // 1. 根据目标类型选择哈希表
    if (dcl_type == LatticeTypes::MoRe) {
        target_hash = g_MoRe_Hash;
    } else if (dcl_type == LatticeTypes::MoMo) {
        target_hash = g_MoMo_Hash;
    } else if (dcl_type == LatticeTypes::Re) {
        target_hash = g_Re_Hash;
    } else if (dcl_type == LatticeTypes::V) {
        target_hash = g_V_Hash;
    }

    if (target_hash == nullptr) {
        return false;
    }

    return hash_set_contains(target_hash, id);
}
//getType重写
template <typename TileType>
__device__ bool tiled_chk_type(long int id, LatticeTypes::lat_type dcl_type, TileType tile, bool active) {
    // LatticeTypes::lat_type final_result = LatticeTypes::Mo; 
    bool final_found = false;
    uint64_t work_mask = tile.ballot(active);
    
    while (work_mask != 0) {
        int leader = __ffsll((unsigned long long int)work_mask) - 1;
        long int target_id = (long int)tile.shfl((unsigned long long int)id, leader);
        // LatticeTypes::lat_type target_type = (LatticeTypes::lat_type)tile.shfl((int)dcl_type, leader);
        LatticeTypes::lat_type target_type = static_cast<LatticeTypes::lat_type>(tile.shfl(static_cast<int>(dcl_type), leader));
        GPUHashSet* target_hash = nullptr;
        if (target_type == LatticeTypes::MoRe) {
            target_hash = g_MoRe_Hash;
        } else if (target_type == LatticeTypes::MoMo) {
            target_hash = g_MoMo_Hash;
        } else if (target_type == LatticeTypes::Re) {
            target_hash = g_Re_Hash;
        } else if (target_type == LatticeTypes::V) {
            target_hash = g_V_Hash;
        }

        bool current_found = false;
        if (target_hash != nullptr) {
          current_found = collective_bucket_contains(target_hash, target_id, tile);
        }

        if (tile.thread_rank() == leader) {
            final_found = current_found;
        }
        
        work_mask &= ~(1ULL << leader);
    }
    return final_found; // 只有原本 active 的线程会拿到有效值
}


template <typename TileType>
__device__ LatticeTypes::lat_type tiled_get_type(long int id, TileType tile, bool active) {
    LatticeTypes::lat_type final_result = LatticeTypes::Mo; 
    uint64_t work_mask = tile.ballot(active);
    
    while (work_mask != 0) {
        int leader = __ffsll((unsigned long long int)work_mask) - 1;
        long int target_id = (long int)tile.shfl((unsigned long long int)id, leader); 
        LatticeTypes::lat_type leader_res = LatticeTypes::Mo;

        if (collective_bucket_contains(g_MoRe_Hash, target_id, tile)) {
            leader_res = LatticeTypes::MoRe;
        } else if (collective_bucket_contains(g_MoMo_Hash, target_id, tile)) {
            leader_res = LatticeTypes::MoMo;
        } else if (collective_bucket_contains(g_Re_Hash, target_id, tile)) {
            leader_res = LatticeTypes::Re;
        } else if (collective_bucket_contains(g_V_Hash, target_id, tile)) {
            leader_res = LatticeTypes::V;
        }

        if (tile.thread_rank() == leader) {
            final_result = leader_res;
        }
        
        work_mask &= ~(1ULL << leader);
    }
    return final_result; // 只有原本 active 的线程会拿到有效值
}
//只计算邻居，不进行哈希查找 getId返回值是-1的话没考虑啊？？
__device__ void get1nn_ids(_type_lattice_coord x, _type_lattice_coord y, _type_lattice_coord z, _type_lattice_id* nn_list) {
  if(x % 2 == 0) {
    nn_list[0] = getId(x - 1, y - 1, z - 1);
    nn_list[1] = getId(x - 1, y - 1, z);
    nn_list[2] = getId(x - 1, y, z - 1);
    nn_list[3] = getId(x - 1, y, z);
    nn_list[4] = getId(x + 1, y - 1, z - 1);
    nn_list[5] = getId(x + 1, y - 1, z);
    nn_list[6] = getId(x + 1, y, z - 1);
    nn_list[7] = getId(x + 1, y, z);
  } else {
    nn_list[0] = getId(x - 1, y, z);
    nn_list[1] = getId(x - 1, y, z + 1);
    nn_list[2] = getId(x - 1, y + 1, z);
    nn_list[3] = getId(x - 1, y + 1, z + 1);
    nn_list[4] = getId(x + 1, y, z);
    nn_list[5] = getId(x + 1, y, z + 1);
    nn_list[6] = getId(x + 1, y + 1, z);
    nn_list[7] = getId(x + 1, y + 1, z + 1);
  }
}
__device__ void get1nn(_type_lattice_coord x, _type_lattice_coord y, _type_lattice_coord z,
                  dev_Lattice *_1nn_list){
  _type_lattice_id latti_id;
  if(x % 2 == 0){
    latti_id = getId(x - 1, y - 1, z - 1);
    getnnlattice(latti_id, &_1nn_list[0]);
    latti_id = getId(x - 1, y - 1, z);
    getnnlattice(latti_id, &_1nn_list[1]);
    latti_id = getId(x - 1, y, z - 1);
    getnnlattice(latti_id, &_1nn_list[2]);
    latti_id = getId(x - 1, y, z);
    getnnlattice(latti_id, &_1nn_list[3]);
    latti_id = getId(x + 1, y - 1, z - 1);
    getnnlattice(latti_id, &_1nn_list[4]);
    latti_id = getId(x + 1, y - 1, z);
    getnnlattice(latti_id, &_1nn_list[5]);
    latti_id = getId(x + 1, y, z - 1);
    getnnlattice(latti_id, &_1nn_list[6]);
    latti_id = getId(x + 1, y, z);
    getnnlattice(latti_id, &_1nn_list[7]);
  }else {
    latti_id = getId(x - 1, y, z);
    getnnlattice(latti_id, &_1nn_list[0]);
    latti_id = getId(x - 1, y, z + 1);
    getnnlattice(latti_id, &_1nn_list[1]);
    latti_id = getId(x - 1, y + 1, z);
    getnnlattice(latti_id, &_1nn_list[2]);
    latti_id = getId(x - 1, y + 1, z + 1);
    getnnlattice(latti_id, &_1nn_list[3]);
    latti_id = getId(x + 1, y, z);
    getnnlattice(latti_id, &_1nn_list[4]);
    latti_id = getId(x + 1, y, z + 1);
    getnnlattice(latti_id, &_1nn_list[5]);
    latti_id = getId(x + 1, y + 1, z);
    getnnlattice(latti_id, &_1nn_list[6]);
    latti_id = getId(x + 1, y + 1, z + 1);
    getnnlattice(latti_id, &_1nn_list[7]);
  }
}
__device__ void get2nn(_type_lattice_coord x, _type_lattice_coord y, _type_lattice_coord z,
                              dev_Lattice *_2nn_list){
  _type_lattice_id latti_id;
  latti_id = getId(x - 2, y, z);
  getnnlattice(latti_id, &_2nn_list[0]);
  // _2nn_list[0] = &_lattices[z + 0][y + 0][x - 2];
  latti_id = getId(x, y - 1, z);
  getnnlattice(latti_id, &_2nn_list[1]);
  // _2nn_list[1] = &_lattices[z + 0][y - 1][x + 0];
  latti_id = getId(x, y, z - 1);
  getnnlattice(latti_id, &_2nn_list[2]);
  // _2nn_list[2] = &_lattices[z - 1][y + 0][x + 0];
  latti_id = getId(x, y, z + 1);
  getnnlattice(latti_id, &_2nn_list[3]);
  // _2nn_list[3] = &_lattices[z + 1][y + 0][x + 0];
  latti_id = getId(x, y + 1, z);
  getnnlattice(latti_id, &_2nn_list[4]);
  // _2nn_list[4] = &_lattices[z + 0][y + 1][x + 0];
  latti_id = getId(x + 2, y, z);
  getnnlattice(latti_id, &_2nn_list[5]);
  // _2nn_list[5] = &_lattices[z + 0][y + 0][x + 2];
}

__device__ int randomInt(unsigned long long seed, int min, int max) {
    // // 获取线程的唯一标识符
    // int threadSeed = threadIdx.x + blockIdx.x * blockDim.x; 

    // // 初始化每个线程的hiprandState
    // hiprandState state;
    // hiprand_init(seed + threadSeed, threadIdx.x, 0, &state); 

    // // 生成一个随机数，范围在[min, max]之间
    // int randNum = hiprand(&state) % (max - min + 1) + min;

    // return randNum;
    return 1;
}

__device__ void getRandomLattice(const _type_lattice_coord& x, const _type_lattice_coord& y, const _type_lattice_coord& z,
                                _type_lattice_coord& temp_x, _type_lattice_coord& temp_y, _type_lattice_coord& temp_z, const int& randomValue){
  // if(randomValue < 0 || randomValue >= 14){
  //   printf("wrong random value:%d\n",randomValue);
  // }
  switch (randomValue)
  {
    case 0:
      if(x % 2 == 0) {
        temp_x = x - 1;
        temp_y = y - 1;
        temp_z = z - 1;
      } else {
        temp_x = x - 1;
        temp_y = y;
        temp_z = z;
      }
      break;
    case 1:
      if(x % 2 == 0) {
        temp_x = x - 1;
        temp_y = y - 1;
        temp_z = z;
      } else {
        temp_x = x - 1;
        temp_y = y;
        temp_z = z + 1;
      }
      break;
    case 2:
      if(x % 2 == 0) {
        temp_x = x - 1;
        temp_y = y;
        temp_z = z - 1;
      } else {
        temp_x = x - 1;
        temp_y = y + 1;
        temp_z = z;
      }
      break;
    case 3:
      if(x % 2 == 0) {
        temp_x = x - 1;
        temp_y = y;
        temp_z = z;
      } else {
        temp_x = x - 1;
        temp_y = y + 1;
        temp_z = z + 1;
      }
      break;
    case 4:
      if(x % 2 == 0) {
        temp_x = x + 1;
        temp_y = y - 1;
        temp_z = z - 1;
      } else {
        temp_x = x + 1;
        temp_y = y;
        temp_z = z;
      }
      break;
    case 5:
      if(x % 2 == 0) {
        temp_x = x + 1;
        temp_y = y - 1;
        temp_z = z;
      } else {
        temp_x = x + 1;
        temp_y = y;
        temp_z = z + 1;
      }
      break;
    case 6:
      if(x % 2 == 0) {
        temp_x = x + 1;
        temp_y = y;
        temp_z = z - 1;
      } else {
        temp_x = x + 1;
        temp_y = y + 1;
        temp_z = z;
      }
      break;
    case 7:
      if(x % 2 == 0) {
        temp_x = x + 1;
        temp_y = y;
        temp_z = z;
      } else {
        temp_x = x + 1;
        temp_y = y + 1;
        temp_z = z + 1;
      }
      break;
    case 8:
      temp_x = x - 2;
      temp_y = y;
      temp_z = z;
      break;
    case 9:
      temp_x = x;
      temp_y = y - 1;
      temp_z = z;
      break;
    case 10:
      temp_x = x;
      temp_y = y;
      temp_z = z - 1;
      break;
    case 11:
      temp_x = x;
      temp_y = y;
      temp_z = z + 1;
      break;
    case 12:
      temp_x = x;
      temp_y = y + 1;
      temp_z = z;
      break;
    case 13:
      temp_x = x + 2;
      temp_y = y;
      temp_z = z;
      break;
    default:
      // printf("wrong random period,%d\n",randomValue);
      // abort();
      break;
  }
}

__device__ float randomFloat(unsigned long long seed) {
    // 获取线程的唯一标识符
    unsigned long long seed1 = clock();
    
    // 定义并初始化hiprandState
    hiprandState state;
    hiprand_init(seed + seed1, threadIdx.x, 0, &state);  // 初始化状态
    
    // 生成 0 到 1 之间的浮点数
    float randNum = hiprand_uniform(&state);  // 使用 hiprand_uniform 生成浮点数
    return randNum;
    // return 0;
}


__device__ void getCoordByLId(_type_lattice_id lid, _type_lattice_coord *x, _type_lattice_coord *y,
                            _type_lattice_coord *z){
   _type_lattice_id lid2 = lid;
    *x = lid % g_meta->size_x;
    lid = lid /  g_meta->size_x;
    *y = lid %  g_meta->size_y;
    *z = lid /  g_meta->size_y;
}
__device__ int isGhostLat(_type_lattice_id lid){
  _type_lattice_coord x = 0, y = 0, z = 0;
  getCoordByLId(lid, &x, &y, &z);
  x -= g_meta->ghost_x;
  y -= g_meta->ghost_y;
  z -= g_meta->ghost_z;

  return x < 0 || y < 0 || z < 0 || x >= g_meta->box_x || y >= g_meta->box_y || z >= g_meta->box_z;

}
// __device__ int isGhostLat(_type_lattice_id lid){
//   _type_lattice_size x = lid%g_meta->size_x;
//   _type_lattice_size y = (lid / g_meta->size_x) % g_meta->size_y;
//   _type_lattice_size z = lid / (g_meta->size_x * g_meta->size_y);
//   if(x<g_meta->ghost_x)
//     return 1;
//   if(x>=g_meta->ghost_x+g_meta->box_x*2)
//     return 1;

//   if(y<g_meta->ghost_y)
//     return 1;
//   if(x>=g_meta->ghost_y+g_meta->box_y)
//     return 1;

//   if(y<g_meta->ghost_z)
//     return 1;
//   if(x>=g_meta->ghost_z+g_meta->box_z)
//     return 1;

//   return 0;
// }

__device__ int isSurfaceLat(_type_lattice_id lid){
   _type_lattice_coord x = 0, y = 0, z = 0;
    getCoordByLId(lid, &x, &y, &z);
    x -= g_meta->ghost_x;
    y -= g_meta->ghost_y;
    z -= g_meta->ghost_z;
    return (0 <= x && x < g_meta->ghost_x) ||(0 <= y && y < g_meta->ghost_y) || (0 <= z && z < g_meta->ghost_z) || (x >= (g_meta->box_x - g_meta->ghost_x) && x < g_meta->box_x) || (y >= (g_meta->box_y - g_meta->ghost_y) && y < g_meta->box_y) || (z >= (g_meta->box_z - g_meta->ghost_z) && z < g_meta->box_z);
}

__global__ void calcExchangePairs_test(int sector_id,int arr_size){
    _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= arr_size)
      return;    

    int flag = false;
    _type_lattice_id id = g_Pair_Atoms[tid];
      // 1. 插入测试
      int test_key = id+1; // 用线程ID作为测试的键
      int insert_result = hash_set_insert_atom_op(g_MoMo_Hash, test_key);
      if (insert_result == 1) {
          printf("Thread %d: Successfully inserted key %d into g_Busy_Set.\n", tid, test_key);
          flag++;
      } else {
          printf("Thread %d: Failed to insert key %d into g_Busy_Set.\n", tid, test_key);
          flag=0;
      }

      // 2. 查找测试
      int contains_result = hash_set_contains(g_MoMo_Hash, test_key);
      if (contains_result == 1) {
          flag++;
          printf("Thread %d: Key %d found in g_Busy_Set.\n", tid, test_key);
      } else {
          flag=0;
          printf("Thread %d: Key %d NOT found in g_Busy_Set.\n", tid, test_key);
      }

      // 3. 删除测试
      int remove_result = hash_set_remove_atom_op(g_MoMo_Hash, test_key);
      if (remove_result == 1) {
          flag++;
          printf("Thread %d: Successfully removed key %d from g_Busy_Set.\n", tid, test_key);
      } else {
          flag=0;
          printf("Thread %d: Failed to remove key %d from g_Busy_Set.\n", tid, test_key);
      }

      // 验证删除后是否仍然存在
      contains_result = hash_set_contains(g_MoMo_Hash, test_key);
      if (contains_result == 1) {
          flag++;
          printf("Thread %d: Key %d still found in g_Busy_Set after removal.\n", tid, test_key);
      } else {
          flag=0;
          printf("Thread %d: Key %d NOT found in g_Busy_Set after removal.\n", tid, test_key);
      }
  }

// __global__ void func_check(int sector_id,int arr_size){
//   _type_lattice_id id = g_Pair_Atoms[0];
//   getType(id);
//   geti
// }

__global__ void update_Pair_Atoms(int arr_size){
    //这个函数用于更新g_pair_atoms
    /*
      一共有两种情况需要更新pair_Atoms
      1,每次calcExchangePairs计算结束后如果发生了交换，则交换的位置要被更新到此函数中
      2,通信过程中需要更新
      这个函数应用于第二种更新，第一种更新直接放到函数中执行就行
    */
    //todo 更新函数有问题
    for(int i=0;i<arr_size;i++)
      g_Pair_Atoms[i] = -1; 
    int index = 0;
    long int momo=0,more=0;
    // printf("-------------MOMO_DATA-------------\n");
    for(int i=0;i<g_MoMo_Hash->capacity;i++){
      if(g_MoMo_Hash->keys[i]!=g_MoMo_Hash->empty_flag && g_MoMo_Hash->keys[i]!=g_MoMo_Hash->tombstone_flag){
        g_Pair_Atoms[index++]=g_MoMo_Hash->keys[i];
        if(!isGhostLat(g_MoMo_Hash->keys[i])){
          // printf("%ld\t",g_MoMo_Hash->keys[i]);
          momo++;
        }
          
      }
    }
    // printf("\n");
    //  printf("-------------MORe_DATA-------------\n");
    for(int i=0;i<g_MoRe_Hash->capacity;i++){
      if(g_MoRe_Hash->keys[i]!=g_MoRe_Hash->empty_flag && g_MoRe_Hash->keys[i]!=g_MoRe_Hash->tombstone_flag){
        g_Pair_Atoms[index++]=g_MoRe_Hash->keys[i];
        if(!isGhostLat(g_MoRe_Hash->keys[i])){
          // printf("%ld\t",g_MoRe_Hash->keys[i]);
          more++;
        }
      }
    }
    // printf("\n");
    // printf("momo num is :%ld\t",momo);
    // printf("more num is:%ld\n",more);
    // printf("AFTER UPDATE atom_num is:%ld/%ld\n",index,arr_size);

}
// Insert-Then-Remove 
template <typename TileType>
__device__ __noinline__ void perform_batch_update_tiled(AtomTx& tx, TileType tile) {
  tx.success = false; // 默认失败
  if (tx.op_type != 0 && tx.src_id == tx.dst_id) {
  
      // if (tile.thread_rank() == 0) {
      //     printf("⚠️ [SELF COLLISION] ... \n");
      // }
      tx.op_type = 0; 
  }
    bool is_type1 = (tx.op_type == 1);
  if (tile.any(is_type1)) {
      // Step 1: 尝试插入 Dst (必须是空位)
      int ins_res = tiled_insert_op(g_MoMo_Hash, tx.dst_id, tile, is_type1,g_MoMo_Hash->num_buckets);
      bool dst_inserted = (is_type1 && ins_res == 1);

      // Step 2: 只有插入成功，才移除 Src
      int rem_res = 0;
      if (tile.any(dst_inserted)) {
          rem_res = tiled_remove_op(g_MoMo_Hash, tx.src_id, tile, dst_inserted,g_MoMo_Hash->num_buckets);
      }
      bool src_removed = (dst_inserted && rem_res == 1);

      // Step 3: 异常处理 (Insert 成功但 Remove 失败 -> 双重存在)
      // 必须回滚 Dst (删除刚刚插入的)
      bool need_rollback = (dst_inserted && !src_removed);
      if (tile.any(need_rollback)) {
        tiled_remove_op(g_MoMo_Hash, tx.dst_id, tile, need_rollback,g_MoMo_Hash->num_buckets);
      }

      // 只有两步都成功才算成功
       if (src_removed){
        tx.success = true;
        // if (tile.thread_rank() == 0) {
        //   printf("[TYPE1_OK] MoMo_src:%ld + dstid:%ld\n", tx.src_id, tx.dst_id);
        // }
      }
  }

  // 2. 处理类型 2：MoMo + Re -> Mo + MoRe (Reaction)
  bool is_type2 = (tx.op_type == 2);
  if (tile.any(is_type2)) {
      // Step 1: 尝试插入 MoRe (Dst) - 这是最关键的一步，如果满/冲突则直接放弃
      int ins_more_res = tiled_insert_op(g_MoRe_Hash, tx.dst_id, tile, is_type2,g_MoRe_Hash->num_buckets);
      bool more_inserted = (is_type2 && ins_more_res == 1);

      // Step 2: 只有插入成功，才去销毁旧原料 (Re 和 MoMo)
      int rem_re_res = 0;
      int rem_momo_res = 0;
      if (tile.any(more_inserted)) {
          rem_re_res = tiled_remove_op(g_Re_Hash, tx.dst_id, tile, more_inserted,g_Re_Hash->num_buckets);
          rem_momo_res = tiled_remove_op(g_MoMo_Hash, tx.src_id, tile, more_inserted,g_MoMo_Hash->num_buckets);
      }
      bool re_removed = (more_inserted && rem_re_res == 1);
      bool momo_removed = (more_inserted && rem_momo_res == 1);

      // Step 3: 提交或回滚
      if (re_removed && momo_removed) {
          tx.success = true;
          // if (tile.thread_rank() == 0) {
          //   printf("[TYPE 2 OK] MoMo_src:%ld + Re_dst:%ld -> MoRe_dst:%ld\n", tx.src_id, tx.dst_id, tx.dst_id);
          // }
      } else {
         // 撤销 MoRe
        bool need_revert_more = more_inserted && !(re_removed && momo_removed);
        if (tile.any(need_revert_more)) {
            int res1 = tiled_remove_op(g_MoRe_Hash, tx.dst_id, tile, need_revert_more,g_MoRe_Hash->num_buckets);
            // 如果撤销插入失败，会导致原子变多 (Duplicate)
            // if (need_revert_more && res1 != 1) {
            //     printf("[DUPLICATE] Type2 Revert MoRe failed at dst_id %ld! (res=%d)\n", tx.dst_id, res1);
            // }
        }
        
        // 撤销误删的 Re
        bool need_restore_re = more_inserted && re_removed && !momo_removed;
        if (tile.any(need_restore_re)) {
            int res2 = tiled_insert_op(g_Re_Hash, tx.dst_id, tile, need_restore_re,g_Re_Hash->num_buckets);
            // if (need_restore_re && res2 != 1) {
            //     printf("[FATAL LEAK] Type2 Restore Re failed at dst_id %ld! Re LOST! (res=%d)\n", tx.dst_id, res2);
            // }
        }
        
        // 撤销误删的 MoMo (这是最容易导致丢失的地方！)
        bool need_restore_momo = more_inserted && !re_removed && momo_removed;
        if (tile.any(need_restore_momo)) {
            int res3 = tiled_insert_op(g_MoMo_Hash, tx.src_id, tile, need_restore_momo,g_MoMo_Hash->num_buckets);
            // if (need_restore_momo && res3 != 1) {
            //     printf("[FATAL LEAK] Type2 Restore MoMo failed at src_id %ld! MoMo LOST! (res=%d). Slot stolen by another thread?\n", tx.src_id, res3);
            // }
        }

       
      }
  }

  // Type 3: MoRe 搬家 (Move to Empty) 
  bool is_type3 = (tx.op_type == 3);
  if (tile.any(is_type3)) {
      // Step 1: Insert Dst
      int ins_res = tiled_insert_op(g_MoRe_Hash, tx.dst_id, tile, is_type3,g_MoRe_Hash->num_buckets);
      bool dst_inserted = (is_type3 && ins_res == 1);
      
      // Step 2: Remove Src
      int rem_res = 0;
      if (tile.any(dst_inserted)) {
          rem_res = tiled_remove_op(g_MoRe_Hash, tx.src_id, tile, dst_inserted,g_MoRe_Hash->num_buckets);
      }
      bool src_removed = (dst_inserted && rem_res == 1);

      // Step 3: Rollback Dst if needed
      bool need_rollback = (dst_inserted && !src_removed);
      if (tile.any(need_rollback)) {
          tiled_remove_op(g_MoRe_Hash, tx.dst_id, tile, need_rollback,g_MoRe_Hash->num_buckets);
      }

      // if (src_removed) tx.success = true;
    if (src_removed){
      tx.success = true;
      // if (tile.thread_rank() == 0) {
      //   printf("[TYPE1_OK] MoMo_src:%ld + dstid:%ld\n", tx.src_id, tx.dst_id);
      // }
    }
  }
  
  // Type 4: MoRe + Re -> Re + MoRe (MoRe 换位)
  bool is_type4 = (tx.op_type == 4);
  if (tile.any(is_type4)) {
      // Step 1: 尝试插入所有新身份 (Re at Src, MoRe at Dst)
      int ins_re_res = tiled_insert_op(g_Re_Hash, tx.src_id, tile, is_type4,g_Re_Hash->num_buckets);
      int ins_more_res = tiled_insert_op(g_MoRe_Hash, tx.dst_id, tile, is_type4,g_MoRe_Hash->num_buckets);
      
      bool re_inserted = (is_type4 && ins_re_res == 1);
      bool more_inserted = (is_type4 && ins_more_res == 1);
      bool both_inserted = (re_inserted && more_inserted);

      // Step 2: 只有都插入成功，才删除旧身份
      int rem_more_res = 0;
      int rem_re_res = 0;
      if (tile.any(both_inserted)) {
          rem_more_res = tiled_remove_op(g_MoRe_Hash, tx.src_id, tile, both_inserted,g_MoRe_Hash->num_buckets);
          rem_re_res = tiled_remove_op(g_Re_Hash, tx.dst_id, tile, both_inserted,g_Re_Hash->num_buckets);
      }
      bool more_removed = (both_inserted && rem_more_res == 1);
      bool re_removed = (both_inserted && rem_re_res == 1);

      // Step 3: 提交或回滚
      if (more_removed && re_removed) {
          tx.success = true;
          // if (tile.thread_rank() == 0) {
          //   printf("[TYPE4OK] Swap MoRe_src:%ld and Re_dst:%ld\n", tx.src_id, tx.dst_id);
          // }
      } else {
        // A. 插入失败的回滚
            bool fail_ins_re = (re_inserted && !more_inserted);
            if (tile.any(fail_ins_re)) {
                tiled_remove_op(g_Re_Hash, tx.src_id, tile, fail_ins_re,g_Re_Hash->num_buckets);
            }
            bool fail_ins_more = (!re_inserted && more_inserted);
            if (tile.any(fail_ins_more)) {
                tiled_remove_op(g_MoRe_Hash, tx.dst_id, tile, fail_ins_more,g_MoRe_Hash->num_buckets);
            }
            
            // B. 删除失败的回滚 (全部回滚)
            bool fail_rem_both = both_inserted && !(more_removed && re_removed);
            if (tile.any(fail_rem_both)) {
                tiled_remove_op(g_Re_Hash, tx.src_id, tile, fail_rem_both,g_Re_Hash->num_buckets);
                tiled_remove_op(g_MoRe_Hash, tx.dst_id, tile, fail_rem_both,g_MoRe_Hash->num_buckets);
            }
            // 恢复误删的 MoRe
        bool need_restore_more = both_inserted && more_removed && !re_removed;
        if (tile.any(need_restore_more)) {
            int res4 = tiled_insert_op(g_MoRe_Hash, tx.src_id, tile, need_restore_more,g_MoRe_Hash->num_buckets);
            // if (need_restore_more && res4 != 1) {
            //     printf("[FATAL LEAK] Type4 Restore MoRe failed at src_id %ld! MoRe LOST! (res=%d)\n", tx.src_id, res4);
            // }
        }

        // 恢复误删的 Re
        bool need_restore_re4 = both_inserted && !more_removed && re_removed;
        if (tile.any(need_restore_re4)) {
            int res5 = tiled_insert_op(g_Re_Hash, tx.dst_id, tile, need_restore_re4,g_Re_Hash->num_buckets);
            // if (need_restore_re4 && res5 != 1) {
            //     printf("[FATAL LEAK] Type4 Restore Re failed at dst_id %ld! Re LOST! (res=%d)\n", tx.dst_id, res5);
            // }
        }

      }
  }
  // 按类别调用独立的函数
  // if (tile.any(tx.op_type == 1)) process_type1(tx, tile);
  // if (tile.any(tx.op_type == 2)) process_type2(tx, tile);
  // if (tile.any(tx.op_type == 3)) process_type3(tx, tile);
  // if (tile.any(tx.op_type == 4)) process_type4(tx, tile);
  if(tx.success){
    //3. 处理 Ghost 和 Surface 更新
    bool update_ghost = isGhostLat(tx.dst_id);
    if (tile.any(update_ghost)) {
        tiled_insert_op(g_ghost_Hash, tx.dst_id, tile, update_ghost,g_ghost_Hash->num_buckets);
    }
    bool update_surface = isSurfaceLat(tx.dst_id);
    if (tile.any(update_surface)) {
        tiled_insert_op(g_surface_Hash, tx.dst_id, tile, update_surface,g_surface_Hash->num_buckets);
    }
    bool update_surface_src = isSurfaceLat(tx.src_id);
    if (tile.any(update_surface_src)) {
        tiled_insert_op(g_surface_Hash, tx.src_id, tile, update_surface_src,g_surface_Hash->num_buckets);
    }
  }
  return;
}

// Insert-Then-Remove 
template <typename TileType>
__device__ __noinline__ void perform_batch_update_single(AtomTx& tx, TileType tile) {
  tx.success = false; // 默认失败
  if (tx.op_type != 0 && tx.src_id == tx.dst_id) {
    tx.op_type = 0; 
  }

  if (tx.op_type == 1) {
    int ins_res = hash_set_insert_atom_op(g_MoMo_Hash, tx.dst_id); 
    if (ins_res == 1) { 
      int rem_res = hash_set_remove_atom_op(g_MoMo_Hash, tx.src_id);     
      if (rem_res == 1) {
          tx.success = true;
      } else {
          hash_set_remove_atom_op(g_MoMo_Hash, tx.dst_id);
      }
    }
  }else if (tx.op_type == 2) {
      int ins_more_res = hash_set_insert_atom_op(g_MoRe_Hash, tx.dst_id);    
      if (ins_more_res == 1) {
        int rem_re_res = hash_set_remove_atom_op(g_Re_Hash, tx.dst_id);
        int rem_momo_res = hash_set_remove_atom_op(g_MoMo_Hash, tx.src_id);

        if (rem_re_res == 1 && rem_momo_res == 1) {
          tx.success = true;
        } else {
            // 回滚
          hash_set_remove_atom_op(g_MoRe_Hash, tx.dst_id);               
          if (rem_re_res == 1) {
              hash_set_insert_atom_op(g_Re_Hash, tx.dst_id);
          }     
          if (rem_momo_res == 1) {
              hash_set_insert_atom_op(g_MoMo_Hash, tx.src_id);
          }
        }
      }
    }else if (tx.op_type == 3) {
        int ins_res = hash_set_insert_atom_op(g_MoRe_Hash, tx.dst_id);
        if (ins_res == 1) {
            int rem_res = hash_set_remove_atom_op(g_MoRe_Hash, tx.src_id);
            if (rem_res == 1) {
                tx.success = true;
            } else {
                hash_set_remove_atom_op(g_MoRe_Hash, tx.dst_id);
            }
        }
    }else if (tx.op_type == 4) {
        int ins_re_res = hash_set_insert_atom_op(g_Re_Hash, tx.src_id);
        int ins_more_res = hash_set_insert_atom_op(g_MoRe_Hash, tx.dst_id);

        if (ins_re_res == 1 && ins_more_res == 1) {
            int rem_more_res = hash_set_remove_atom_op(g_MoRe_Hash, tx.src_id);
            int rem_re_res = hash_set_remove_atom_op(g_Re_Hash, tx.dst_id);

            if (rem_more_res == 1 && rem_re_res == 1) {
                tx.success = true;
            } else {
                hash_set_remove_atom_op(g_Re_Hash, tx.src_id);
                hash_set_remove_atom_op(g_MoRe_Hash, tx.dst_id);
                if (rem_more_res == 1) hash_set_insert_atom_op(g_MoRe_Hash, tx.src_id);
                if (rem_re_res == 1) hash_set_insert_atom_op(g_Re_Hash, tx.dst_id);
            }
        } else {
          if (ins_re_res == 1) hash_set_remove_atom_op(g_Re_Hash, tx.src_id);
          if (ins_more_res == 1) hash_set_remove_atom_op(g_MoRe_Hash, tx.dst_id);
        }
    }


  if (tx.success) {
    if (isGhostLat(tx.dst_id)) {
        hash_set_insert_atom_op(g_ghost_Hash, tx.dst_id);
    }
    if (isSurfaceLat(tx.dst_id)) {
        hash_set_insert_atom_op(g_surface_Hash, tx.dst_id);
    }
    if (isSurfaceLat(tx.src_id)) {
        hash_set_insert_atom_op(g_surface_Hash, tx.src_id);
    }
  }
  return;
}

__device__ int decide_op_type(LatticeTypes::lat_type cur_type,LatticeTypes::lat_type des_type,bool flag) {
    if (cur_type == LatticeTypes::MoMo) {
      if (des_type == LatticeTypes::Mo) return 1; // 交换位置 
      if (des_type == LatticeTypes::Re) return 2; // (momo 变成 mo，re 变成 more)
    }
    if (cur_type == LatticeTypes::MoRe) {
      if (flag && des_type == LatticeTypes::Mo) return 3; // MoRe 搬家
      if (flag && des_type == LatticeTypes::Re) return 4; // MoRe 换位
    }
    return 0; // 不满足条件
}

// template <typename TileType>
// __device__ bool collective_bucket_contains(GPUHashSet* set, long int target_key, TileType tile) {
//     unsigned int buck_num = set->hasher(target_key) % set->num_buckets;
//     int lane = tile.thread_rank();
//     int loop_count = 0;

//     // 必须加上线性探测的循环，处理桶溢出的情况
//     while (loop_count < set->num_buckets) {
//         // 全组 64 线程同时加载整个桶
//         long int val = atomic_load_relaxed(&set->keys[buck_num * 64 + lane]);

//         if (tile.any(val == target_key)) {
//             return true; 
//         }
//         if (tile.any(val == set->empty_flag)) {
//             return false;
//         }
//         buck_num = (buck_num + 1) % set->num_buckets;
//         loop_count++;
//     }
//     return false;
// }

template <typename TileType>
__device__ bool collective_bucket_contains(GPUHashSet* set, long int target_key, TileType tile) {
    // 即使 target_key <= 0，也不能提前 return，必须陪全组参与 tile.any 防死锁
    if (set == nullptr) return false;

    const bool valid_key = (target_key > 0);
    const unsigned int max_probe = hash_effective_max_probes(set->num_buckets);
    unsigned int loop_count = 0;
    unsigned int buck_num = valid_key ? hash_probe_bucket(set->num_buckets, target_key, loop_count) : 0;
    const int lane = tile.thread_rank();
    bool done = !valid_key;
    bool result = false;
    // const int MAX_PROBE_STEPS = 3;

    while (loop_count <= max_probe) {
        long int val = GPU_HASH_EMPTY_KEY;
        if (!done) {
            val = atomic_load_relaxed(&set->keys[buck_num * BUCKET_SIZE + lane]);
        }

        const uint64_t active_mask = tile.ballot(!done);
        if (active_mask == 0) break;

        const uint64_t match_mask = tile.ballot(!done && val == target_key);
        const uint64_t empty_mask = tile.ballot(!done && val == GPU_HASH_EMPTY_KEY);

        if (match_mask != 0) {
            result = true;
            done = true;
        } else if (empty_mask != 0) {
            result = false;
            done = true;
        } else if (!done) {
            loop_count++;
            if (loop_count > max_probe) {
                result = false;
                done = true;
            } else {
                buck_num = hash_probe_bucket(set->num_buckets, target_key, loop_count);
            }
        }
    }
    return result;
}



template <typename TileType>
__device__ void tiled_check_neighbors(_type_lattice_size x, _type_lattice_size y, _type_lattice_size z, 
                                      TileType tile, bool active, int& out_numSIA, int& out_numRe) 
{
    out_numSIA = 0;
    out_numRe = 0;
    uint64_t work_mask = tile.ballot(active);

    while (work_mask != 0) {
        int leader = __ffsll((unsigned long long int)work_mask) - 1;
        
        _type_lattice_size lx = tile.shfl(x, leader);
        _type_lattice_size ly = tile.shfl(y, leader);
        _type_lattice_size lz = tile.shfl(z, leader);
        
        _type_lattice_id nn_ids[8];
        get1nn_ids(lx, ly, lz, nn_ids); 

        int leader_SIA = 0;
        int leader_Re = 0;
        #pragma unroll 1 
        for (int i = 0; i < 8; i++) {
            _type_lattice_id nid = nn_ids[i];
           
            if (collective_bucket_contains(g_MoRe_Hash, nid, tile) || collective_bucket_contains(g_MoMo_Hash, nid, tile)) {
                leader_SIA++;
            } 
            else if (collective_bucket_contains(g_Re_Hash, nid, tile)) {
                leader_Re++;
            }
        }

        if (tile.thread_rank() == leader) {
            out_numSIA = leader_SIA;
            out_numRe = leader_Re;
        }
        work_mask &= ~(1ULL << leader);
    }
}
template <typename TileType>
__device__ void smart_tiled_check_neighbors(_type_lattice_size x, _type_lattice_size y, _type_lattice_size z, 
                                            TileType tile, bool active, int& out_numSIA, int& out_numRe) 
{
    out_numSIA = 0;
    out_numRe = 0;
    
    uint64_t work_mask = tile.ballot(active);
    
    // 统计当前 64 个人里，到底有几个 active 的？(__popcll 是极速硬件指令)
    int active_count = __popcll((unsigned long long int)work_mask);

    // 设置临界值 THRESHOLD// 通常在 3 到 6 之间
    const int THRESHOLD = 4; 

    if (active_count > 0 && active_count <= THRESHOLD) {
        while (work_mask != 0) {
            int leader = __ffsll((unsigned long long int)work_mask) - 1;
            work_mask &= ~(1ULL << leader);
        }
    } 
    else if (active_count > THRESHOLD) {
        if (active) {
            _type_lattice_id nn_ids[8];
            get1nn_ids(x, y, z, nn_ids); 
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                _type_lattice_id nid = nn_ids[i];
                if (nid < 0) continue;
                if (hash_set_contains(g_MoRe_Hash, nid) || hash_set_contains(g_MoMo_Hash, nid)) out_numSIA++;
                else if (hash_set_contains(g_Re_Hash, nid)) out_numRe++;
            }
        }
    }
}
__device__ void tiled_check_neighbors_single(_type_lattice_size x, _type_lattice_size y, _type_lattice_size z, int& out_numSIA, int& out_numRe) 
{
    out_numSIA = 0;
    out_numRe = 0;

    _type_lattice_id nn_ids[8];
    get1nn_ids(x, y, z, nn_ids); 

    for (int i = 0; i < 8; i++) {
        _type_lattice_id nid = nn_ids[i];
        if (nid < 0) continue;
        
        if (hash_set_contains(g_MoRe_Hash, nid) || hash_set_contains(g_MoMo_Hash, nid)) {
            out_numSIA++;
        } else if (hash_set_contains(g_Re_Hash, nid)) {
            out_numRe++;
        }
    }
}
__global__ void calcExchangePairs(int sector_id,int arr_size, RecbVerifyStats *stats
){
  #if KMC_RECB_DEVICE_PRINTF
  printf("begin calcExchangePairs\n");
#endif
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<KMC_HASH_TILE_SIZE>(block);
  // 初始化状态机
  AtomTx tx = {0, 0, true, false, 0};

  _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
  _type_lattice_id id = 0;

  bool valid_thread = (g_Pair_Atoms != nullptr && tid < arr_size);
  if (valid_thread) {
    id = g_Pair_Atoms[tid];
  }

  if(!valid_thread || id<=0){
    tx.active = false;
    id = 0;
  }

  //====================tile一组线程同时查哈希表======================
  // LatticeTypes::lat_type cur_type = LatticeTypes::Mo;
  // LatticeTypes::lat_type cur_type = tiled_get_type(id, tile, tx.active);
  // if(tx.active){
  //   printf("[Thread %d] My ID is %ld, and my cur_type is: %d\n", threadIdx.x, id, cur_type);
  // }
   LatticeTypes::lat_type cur_type =tx.active? getType(id):(LatticeTypes::Mo);//单线程查
  //  LatticeTypes::lat_type cur_type = tiled_get_type(id, tile, tx.active);

  
  // if (tx.active) {
  //     printf("[GPU Check] Thread %d processing ID: %ld. Found Type: %d\n", threadIdx.x, id, cur_type);
  // }

  //================================================================
  _type_lattice_size x = 0;
  _type_lattice_size y = 0;
  _type_lattice_size z = 0;

  if (tx.active) {
      // cur_type = getType(id);
      x = id % g_meta->size_x;
      y = (id / g_meta->size_x) % g_meta->size_y;
      z = id / (g_meta->size_x * g_meta->size_y);
      tx.src_id = id;
  }
  //==========================================
  const long int x_min = 2*g_meta->regions[sector_id * 6 + 0];
  const long int x_max = 2*g_meta->regions[sector_id * 6 + 1];
  const long int y_min = g_meta->regions[sector_id * 6 + 2];
  const long int y_max = g_meta->regions[sector_id * 6 + 3];
  const long int z_min = g_meta->regions[sector_id * 6 + 4];
  const long int z_max = g_meta->regions[sector_id * 6 + 5];

  int my_hops = 0;
  int started_in_sector = 0;
  int captured_no_move = 0;
  if (tx.active) {
    const bool in0 = (x_min <= x && x < x_max &&
                      y_min <= y && y < y_max &&
                      z_min <= z && z < z_max);
    if (in0) {
      started_in_sector = 1;
      if (stats) atomicAdd(&stats->sector_pairs, 1u);
    }
  }

  _type_lattice_size temp_x, temp_y, temp_z; 
  int loop_index = 0;

  while (true) { 
        loop_index++;
        
        // 统一更新出界状态
        if (tx.active) {
          bool in_bounds = (x_min <= x && x < x_max &&
                            y_min <= y && y < y_max &&
                            z_min <= z && z < z_max);
          if (!in_bounds) {
            tx.active = false;
          }
        }

        //Tile 同步：如果没有需要工作的线程，统一跳出
        uint64_t work_mask = tile.ballot(tx.active);
        if (work_mask == 0) break;

        // --- 全组查邻居 ---
        int numSIA = 0, numRe = 0;
        // tiled_check_neighbors(x, y, z, tile, tx.active, numSIA, numRe);

        if (tx.active) {
          tiled_check_neighbors_single(x, y, z, numSIA, numRe); //单线程查
          // printf("[checkThread %d]numSIA= %d, numRe=%d,and my cur_type is: %d\n", threadIdx.x, numSIA, numRe,cur_type);
          if (numSIA >= 2) {
            if (my_hops == 0) captured_no_move = 1;
            tx.active = false;
          } else if (cur_type == LatticeTypes::MoRe && numRe >= 5) {
            if (my_hops == 0) captured_no_move = 1;
            tx.active = false;
          }
        }
       
        // --- 计算随机跳跃目标 ---
        _type_lattice_size temp_x = 0, temp_y = 0, temp_z = 0;
        _type_lattice_id random_id = 0;
        if (tx.active) {
            int sum = tid + id + loop_index;
            int randomNumber = (sum % 14 + 14) % 14;
            getRandomLattice(x, y, z, temp_x, temp_y, temp_z, randomNumber);
            random_id = getId(temp_x, temp_y, temp_z);
        }
        LatticeTypes::lat_type rand_type = tx.active? getType(random_id):(LatticeTypes::Mo);

        if (tx.active) {
            if (rand_type == LatticeTypes::Mo || rand_type == LatticeTypes::Re) {
                bool flag1 = (numRe == 0);
                bool flag2 = false;
                if (numRe > 0 && numRe < LatticesList::MAX_1NN) {
                    double ranmov = randomFloat(tid + id + loop_index);
                    flag2 = (ranmov > (static_cast<double>(numRe) / 8.0));
                }
                tx.op_type = decide_op_type(cur_type, rand_type, flag1 || flag2);
                tx.dst_id = random_id;
            } else {
                tx.op_type = 0;
            }
        } else {
            tx.op_type = 0;
        }
        resolve_intra_tile_conflicts(tx, tile);
        //----加锁----
        bool locks_acquired = false;
        if (tx.active && tx.op_type != 0) {
            locks_acquired = try_acquire_bitmap_locks(tx.src_id, tx.dst_id);
            if (!locks_acquired) {
              // 如果没抢到物理锁，把 op_type 设为 0
              tx.op_type = 0; 
            }
        }
        //  DCL 状态二次校验 (只有手里有锁，并且想干活的人才做)
        bool need_verify = locks_acquired && (tx.op_type != 0);

        // 用最新的状态，再去查一次老家和目的地！
        // LatticeTypes::lat_type safe_src_type = tiled_get_type(tx.src_id, tile, need_verify);
        // LatticeTypes::lat_type safe_dst_type = tiled_get_type(tx.dst_id, tile, need_verify);
        //------------tile协作版---------------
        bool src_chk = tiled_chk_type(tx.src_id, cur_type, tile, need_verify);
        bool dst_chk = tiled_chk_type(tx.dst_id, rand_type, tile, need_verify);

        //-------------单线程版----------------
        // bool src_chk = single_chk_type(tx.src_id, cur_type, tile, need_verify);
        // bool dst_chk = single_chk_type(tx.dst_id, rand_type, tile, need_verify);
        
        // LatticeTypes::lat_type safe_src_type = need_verify?getType(tx.src_id):(LatticeTypes::Mo);
        // LatticeTypes::lat_type safe_dst_type = need_verify?getType(tx.dst_id):(LatticeTypes::Mo);

        if (need_verify) {          
          if (!src_chk || !dst_chk) {
            tx.op_type = 0;
          }
        }
        //---- 核心事务批量插入删除操作 (全组进入) ---
        perform_batch_update_tiled(tx, tile);
        //----解锁----
        if (locks_acquired) {
            release_bitmap_locks(tx.src_id, tx.dst_id);
        }
        //----更新状态----
        // 对齐 original recb_solver：成功换格后继续在本 while 里跳，直到抓住或出扇区。
        // 抓住 / 出界只关 active，全组走完 tile 集体操作后由 ballot 一起 break，不能 return。
        if (tx.active && tx.op_type != 0 && tx.success) {
            x = temp_x;
            y = temp_y;
            z = temp_z;
            tx.src_id = id = random_id;
            if(tx.op_type == 2 || tx.op_type == 4) cur_type = LatticeTypes::MoRe;
            my_hops++;
            const bool dest_in_sector = (x_min <= x && x < x_max &&
                                         y_min <= y && y < y_max &&
                                         z_min <= z && z < z_max);
            if (dest_in_sector && !isGhostLat(random_id)) {
              g_Pair_Atoms[tid] = random_id;
            } else {
              g_Pair_Atoms[tid] = -1;
              tx.active = false;
            }
        }
        tx.op_type = 0;
    }
    if (stats && started_in_sector) {
      if (my_hops > 0) atomicAdd(&stats->hops, (unsigned long long)my_hops);
      atomicMax(&stats->max_hops, (unsigned int)my_hops);
      if (my_hops >= 2) atomicAdd(&stats->multi_hop_pairs, 1u);
      if (captured_no_move) atomicAdd(&stats->captured_no_move, 1u);
    }
}

__global__ void calcExchangePairsTemp(int sector_id,int arr_size
){
  // print_g_Pair_Atoms(arr_size);
  // checkpointers();
  if(g_Pair_Atoms == nullptr)
    return;
  _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= arr_size)
    return;
  // if(tid >= 1)
  //   return;
  
  _type_lattice_id id = g_Pair_Atoms[tid];
  if(id <= 0)
    return;
  LatticeTypes::lat_type cur_type = getType(id);
  
  _type_lattice_size x = id%g_meta->size_x;
  _type_lattice_size y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_size z = id / (g_meta->size_x * g_meta->size_y);
  const long int x_min = 2*g_meta->regions[sector_id * 6 + 0];
  const long int x_max = 2*g_meta->regions[sector_id * 6 + 1];
  const long int y_min = g_meta->regions[sector_id * 6 + 2];
  const long int y_max = g_meta->regions[sector_id * 6 + 3];
  const long int z_min = g_meta->regions[sector_id * 6 + 4];
  const long int z_max = g_meta->regions[sector_id * 6 + 5];
  // printf("id %ld, cur_type is %ld,(%ld,%ld,%ld) -----(%ld-%ld)(%ld-%ld)(%ld-%ld)\n",id,cur_type,x,y,z,x_min,x_max,y_min,y_max,z_min,z_max);

  //if not in this domain then skip

   // 定义区域边界
   

    // 检查每个维度的边界条件
    bool x_ok = (x_min <= x && x < x_max);
    bool y_ok = (y_min <= y && y < y_max);
    bool z_ok = (z_min <= z && z < z_max);

    // // 输出调试信息
  
  // if(!(2 * g_meta->regions[sector_id * 6 + 0] <= x && x < 2*g_meta->regions[sector_id * 6 + 1] && g_meta->regions[sector_id * 6 + 2] <= y && y < g_meta->regions[sector_id * 6 + 3] && g_meta->regions[sector_id * 6 + 4] <= z && z < g_meta->regions[sector_id * 6 + 5])){
  // //   // // // 输出调试信息
  //   // printf("id %ld: x_ok=%d, y_ok=%d, z_ok=%d\n", 
  //   //       id, x_ok, y_ok, z_ok);
  //   return;
  // }
    // printf("cal atom msg:\n");s
    // printf("Sector %d: x=[%ld,%ld), y=[%ld,%ld), z=[%ld,%ld)\t", 
    //       sector_id, x_min, x_max, y_min, y_max, z_min, z_max);
    // printf("Point %ld:(%ld,%ld,%ld): %d \n", id,
    //       x, y, z,getType(id));


  
  int numSIA=0;
  int numRe=0;
  int ret = 0;

  int num_more = 0;
  int num_momo = 0;
  int busy = 0;

  for(int i =0;i<g_MoRe_Hash->capacity;i++){
    if(g_MoRe_Hash->keys[i]!=-1&&g_MoRe_Hash->keys[i]!=-2)
      num_more++;
  }

  for(int i =0;i<g_MoMo_Hash->capacity;i++){
    if(g_MoMo_Hash->keys[i]!=-1&&g_MoMo_Hash->keys[i]!=-2)
      num_momo++;
  }

  // printf("begin %d + %d = %d \n",num_more,num_momo,num_momo+num_more);
  dev_Lattice nn_list[LatticesList::MAX_1NN];
  _type_lattice_size temp_x, temp_y, temp_z;
  dev_Lattice randomLattice;
  int loop_index = 0;
  while (2 * g_meta->regions[sector_id * 6 + 0] <= x && x < 2*g_meta->regions[sector_id * 6 + 1] && g_meta->regions[sector_id * 6 + 2] <= y && y < g_meta->regions[sector_id * 6 + 3] && g_meta->regions[sector_id * 6 + 4] <= z && z < g_meta->regions[sector_id * 6 + 5]) 
  {

    // printf("id:%ld enter loop:%d\n",id,loop_index);
    loop_index++;
    if(loop_index >=10)
      break;

    id = getId(x,y,z);

    if(cur_type != getType(id)){
      // printf("%d:%d id has wrong type %d - %d\n",tid,id,cur_type,getType(id));
    }

 
    if(hash_set_contains(g_Busy_Set,id)){
      continue;
    }else{
      ret = hash_set_insert_atom_op(g_Busy_Set,id);
      busy++;
    }
    numSIA = 0;
    get1nn(x,y,z,nn_list);
    for(int i=0;i < LatticesList::MAX_1NN;i++){
      if(nn_list[i].type == LatticeTypes::MoMo || nn_list[i].type == LatticeTypes::MoRe)
        numSIA++;
    }

    if(numSIA >= 2) {
      ret = hash_set_remove_atom_op(g_Busy_Set,id);
      if(ret == 1)
      busy--;
      return;
    }
    numRe = 0;
    
    if (cur_type == LatticeTypes::MoRe) {
        for(int i=0;i < LatticesList::MAX_1NN;i++) {
          if (nn_list[i].type == LatticeTypes::Re) numRe++;
        }
        if (numRe >= 5) {
          ret = hash_set_remove_atom_op(g_Busy_Set,id);
          if(ret == 1)
          busy--;
          return;
        }
    }

    int sum = tid + id + loop_index;
    int randomNumber = (sum % 14 + 14) % 14;  // 确保结果在0-13之间

    if(randomNumber < 0 || randomNumber >= 14){
        printf("%ld : random num generate error %d\n",(long)tid,randomNumber);
      }
    getRandomLattice(x,y,z,temp_x,temp_y,temp_z,randomNumber);
    randomLattice.id = getId(temp_x,temp_y,temp_z);

    
    if(hash_set_contains(g_Busy_Set,randomLattice.id)){
      ret = hash_set_remove_atom_op(g_Busy_Set,id);
      if(ret == 1)
      busy--;
      continue;
    }else{
      ret = hash_set_insert_atom_op(g_Busy_Set,randomLattice.id);
      busy++;
    }
    // printf("%d:is finish select random number\n",tid);
    randomLattice.type = getType(randomLattice.id);
    LatticeTypes::lat_type rantype = randomLattice.type;

    // printf("%ld --- %ld(%ld,%ld,%ld)->%ld(%ld,%ld,%ld),\t%d->%d\n",tid,id,x,y,z,randomLattice.id,temp_x,temp_y,temp_z,cur_type,randomLattice.type);


    if(randomLattice.type == LatticeTypes::Mo || randomLattice.type == LatticeTypes::Re){
      
      if(cur_type == LatticeTypes::MoMo){
        bool exchanged = false;
        if (randomLattice.type == LatticeTypes::Mo) {
          int rm = hash_set_remove_atom_op(g_MoMo_Hash,id);
          if (rm != 1) {
            ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
            if(ret == 1) busy--;
            ret = hash_set_remove_atom_op(g_Busy_Set,id);
            if(ret == 1) busy--;
            continue;
          }
          int ins = hash_set_insert_atom_op(g_MoMo_Hash,randomLattice.id);
          if (ins != 1) {
            hash_set_insert_atom_op(g_MoMo_Hash,id);
            ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
            if(ret == 1) busy--;
            ret = hash_set_remove_atom_op(g_Busy_Set,id);
            if(ret == 1) busy--;
            continue;
          }
          exchanged = true;
        } else if(randomLattice.type == LatticeTypes::Re) {
          int rm_re = hash_set_remove_atom_op(g_Re_Hash,randomLattice.id);
          if (rm_re != 1) {
            ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
            if(ret == 1) busy--;
            ret = hash_set_remove_atom_op(g_Busy_Set,id);
            if(ret == 1) busy--;
            continue;
          }
          int rm_momo = hash_set_remove_atom_op(g_MoMo_Hash,id);
          if (rm_momo != 1) {
            hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
            ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
            if(ret == 1) busy--;
            ret = hash_set_remove_atom_op(g_Busy_Set,id);
            if(ret == 1) busy--;
            continue;
          }
          int ins_more = hash_set_insert_atom_op(g_MoRe_Hash,randomLattice.id);
          if (ins_more != 1) {
            hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
            hash_set_insert_atom_op(g_MoMo_Hash,id);
            ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
            if(ret == 1) busy--;
            ret = hash_set_remove_atom_op(g_Busy_Set,id);
            if(ret == 1) busy--;
            continue;
          }
          cur_type = LatticeTypes::MoRe;
          exchanged = true;
        }

        if (!exchanged) {
          ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
          if(ret == 1) busy--;
          ret = hash_set_remove_atom_op(g_Busy_Set,id);
          if(ret == 1) busy--;
          continue;
        }

        x = temp_x;
        y = temp_y;
        z = temp_z;
        
        if (isGhostLat(randomLattice.id)) {
          ret = hash_set_insert_atom_op(g_ghost_Hash,randomLattice.id);
        }
        if (isSurfaceLat(randomLattice.id)){
          ret = hash_set_insert_atom_op(g_surface_Hash,randomLattice.id);
        }
          
        if (isSurfaceLat(id)){
          ret = hash_set_insert_atom_op(g_surface_Hash,id);
        }
      }
      else if(cur_type == LatticeTypes::MoRe){
        if(numRe == 0){
          bool exchanged = false;
          if (randomLattice.type == LatticeTypes::Mo) {
            int rm_more = hash_set_remove_atom_op(g_MoRe_Hash,id);
            if (rm_more != 1) {
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            int ins_more = hash_set_insert_atom_op(g_MoRe_Hash,randomLattice.id);
            if (ins_more != 1) {
              hash_set_insert_atom_op(g_MoRe_Hash,id);
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            exchanged = true;
          } else {
            int rm_re = hash_set_remove_atom_op(g_Re_Hash,randomLattice.id);
            if (rm_re != 1) {
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            int rm_more = hash_set_remove_atom_op(g_MoRe_Hash,id);
            if (rm_more != 1) {
              hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            int ins_re = hash_set_insert_atom_op(g_Re_Hash,id);
            if (ins_re != 1) {
              hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
              hash_set_insert_atom_op(g_MoRe_Hash,id);
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            int ins_more = hash_set_insert_atom_op(g_MoRe_Hash,randomLattice.id);
            if (ins_more != 1) {
              hash_set_remove_atom_op(g_Re_Hash,id);
              hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
              hash_set_insert_atom_op(g_MoRe_Hash,id);
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            exchanged = true;
          }

          if (!exchanged) {
            ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
            if(ret == 1) busy--;
            ret = hash_set_remove_atom_op(g_Busy_Set,id);
            if(ret == 1) busy--;
            continue;
          }
          x = temp_x;
          y = temp_y;
          z = temp_z;
          // printf("2 current type %d\n",getType(getId(temp_x,temp_y,temp_z)));

          //TODO:
          if (isGhostLat(randomLattice.id)) {
            ret = hash_set_insert_atom_op(g_ghost_Hash,randomLattice.id);
          }
          if (isSurfaceLat(randomLattice.id))
            ret = hash_set_insert_atom_op(g_surface_Hash,randomLattice.id);
          if (isSurfaceLat(id)) 
            ret = hash_set_insert_atom_op(g_surface_Hash,id);
        }
        else if(numRe > 0 && numRe < LatticesList::MAX_1NN){
          double ranmov = randomFloat(tid + id + loop_index);
          if(ranmov > (static_cast<double>(numRe) / LatticesList::MAX_1NN)) {
            bool exchanged = false;
            if (randomLattice.type == LatticeTypes::Mo) {
              int rm_more = hash_set_remove_atom_op(g_MoRe_Hash,id);
              if (rm_more != 1) {
                ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
                if(ret == 1) busy--;
                ret = hash_set_remove_atom_op(g_Busy_Set,id);
                if(ret == 1) busy--;
                continue;
              }
              int ins_more = hash_set_insert_atom_op(g_MoRe_Hash,randomLattice.id);
              if (ins_more != 1) {
                hash_set_insert_atom_op(g_MoRe_Hash,id);
                ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
                if(ret == 1) busy--;
                ret = hash_set_remove_atom_op(g_Busy_Set,id);
                if(ret == 1) busy--;
                continue;
              }
              exchanged = true;
            } else if(randomLattice.type == LatticeTypes::Re) {
              int rm_re = hash_set_remove_atom_op(g_Re_Hash,randomLattice.id);
              if (rm_re != 1) {
                ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
                if(ret == 1) busy--;
                ret = hash_set_remove_atom_op(g_Busy_Set,id);
                if(ret == 1) busy--;
                continue;
              }
              int rm_more = hash_set_remove_atom_op(g_MoRe_Hash,id);
              if (rm_more != 1) {
                hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
                ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
                if(ret == 1) busy--;
                ret = hash_set_remove_atom_op(g_Busy_Set,id);
                if(ret == 1) busy--;
                continue;
              }
              int ins_re = hash_set_insert_atom_op(g_Re_Hash,id);
              if (ins_re != 1) {
                hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
                hash_set_insert_atom_op(g_MoRe_Hash,id);
                ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
                if(ret == 1) busy--;
                ret = hash_set_remove_atom_op(g_Busy_Set,id);
                if(ret == 1) busy--;
                continue;
              }
              int ins_more = hash_set_insert_atom_op(g_MoRe_Hash,randomLattice.id);
              if (ins_more != 1) {
                hash_set_remove_atom_op(g_Re_Hash,id);
                hash_set_insert_atom_op(g_Re_Hash,randomLattice.id);
                hash_set_insert_atom_op(g_MoRe_Hash,id);
                ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
                if(ret == 1) busy--;
                ret = hash_set_remove_atom_op(g_Busy_Set,id);
                if(ret == 1) busy--;
                continue;
              }
              exchanged = true;
            } else {
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }

            if (!exchanged) {
              ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
              if(ret == 1) busy--;
              ret = hash_set_remove_atom_op(g_Busy_Set,id);
              if(ret == 1) busy--;
              continue;
            }
            x = temp_x;
            y = temp_y;
            z = temp_z;
            // printf("3 current type %d\n",getType(getId(temp_x,temp_y,temp_z)));
            //TODO
            if (isGhostLat(randomLattice.id)) {
              ret = hash_set_insert_atom_op(g_ghost_Hash,randomLattice.id);
            }
            if (isSurfaceLat(randomLattice.id)) 
              ret = hash_set_insert_atom_op(g_surface_Hash,randomLattice.id);
            if (isSurfaceLat(id)) 
              ret = hash_set_insert_atom_op(g_surface_Hash,id);
         }else continue;
      }
        else{
          ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
          if(ret == 1)
          busy--;
          ret = hash_set_remove_atom_op(g_Busy_Set,id);
          if(ret == 1)
          busy--;
          // printf("%d : more : %d momo : %d \n",tid,num_more,num_momo);
          return;
        } 
    }
    else {
      ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
      if(ret == 1)
        busy--;
      ret = hash_set_remove_atom_op(g_Busy_Set,id);
      if(ret == 1)
        busy--;
      continue;
      }
    }  
    ret = hash_set_remove_atom_op(g_Busy_Set,randomLattice.id);
    if(ret == 1)
    busy--;
    ret = hash_set_remove_atom_op(g_Busy_Set,id);
    if(ret == 1)
    busy--;

    
  }

}

// 测试内核
__global__ void test_kernel(GPUHashSet* gpu_set) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < gpu_set->capacity) {
      int key = gpu_set->keys[tid];
      if (key != gpu_set->empty_flag) {
          printf("Key at index %d: %d\n", tid, key);
      }
  }
}

__global__ void get_surface_and_ghost_type(long int *surface_type,long int *ghost_type){
      _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
      if(tid == 0){
        // printf("tp capacity:%ld\n",g_surface_Hash->capacity);
        for(int i=0;i<g_surface_Hash->capacity;i++){
          if(g_surface_Hash->keys[i] != -1 && g_surface_Hash->keys[i] != -2){
            // printf("tp2 %ld\t", g_surface_Hash->keys[i]);
            surface_type[i] = getType(g_surface_Hash->keys[i]);
            // printf("tp %ld\t", getType(g_surface_Hash->keys[i]));
          }
            
        }
        //  printf("\n");
      }else if(tid == 1){
        // printf("tp capacity:sld\n",g_ghost_Hash->capacity);
        for(int i=0;i<g_ghost_Hash->capacity;i++){
          if(g_ghost_Hash->keys[i] != -1 && g_ghost_Hash->keys[i] != -2){
            // printf("tp3 %ld\t", g_ghost_Hash->keys[i]);
            ghost_type[i] = getType(g_ghost_Hash->keys[i]);
            // printf("tp1 %ld\t", getType(g_ghost_Hash->keys[i]));
          }
            
        }
        // printf("\n");
    }
}
__global__ void compute_types_for_keys(const long int* __restrict__ keys,
                                      long int* __restrict__ out_types,
                                      int capacity) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= capacity) return;
    long int k = keys[tid];
    if (k != -1 && k != -2) out_types[tid] = getType(k);
}
__global__ void compact_valid_keys(const long int* __restrict__ keys,
                                  int capacity,
                                  long int* __restrict__ out_keys,
                                  int* __restrict__ out_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= capacity) return;
    long int k = keys[tid];
    if (k == -1 || k == -2) return;
    int idx = atomicAdd(out_count, 1);
    out_keys[idx] = k;
}

__global__ void filter_ids_in_region(const long int* __restrict__ in_ids,
                                     int in_count,
                                     int x_low, int x_high,
                                     int y_low, int y_high,
                                     int z_low, int z_high,
                                     long int* __restrict__ out_ids,
                                     int* __restrict__ out_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= in_count) return;
  long int id = in_ids[tid];
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  if (x_low <= x && x < x_high && y_low <= y && y < y_high && z_low <= z && z < z_high) {
    int idx = atomicAdd(out_count, 1);
    out_ids[idx] = id;
  }
}

__global__ void build_nn_info_from_vac_list(const long int* __restrict__ vac_ids,
                                           int vac_count,
                                           dev_Vacancy* __restrict__ out_vac,
                                           dev_nnLattice* __restrict__ out_nn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vac_count) return;
  long int vid = vac_ids[i];
  _type_lattice_id x = vid % g_meta->size_x;
  _type_lattice_id y = (vid / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = vid / (g_meta->size_x * g_meta->size_y);
  out_vac[i].id = static_cast<int>(vid);

  dev_Lattice nn1[8];
  dev_Lattice nn2[6];
  get1nn(x, y, z, nn1);
  get2nn(x, y, z, nn2);

  int base = i * NN_TOTAL;
  for (int j = 0; j < 8; j++) {
    out_nn[base + j].id = static_cast<int>(nn1[j].id);
    out_nn[base + j].type._type = nn1[j].type;
  }
  for (int j = 0; j < 6; j++) {
    out_nn[base + 8 + j].id = static_cast<int>(nn2[j].id);
    out_nn[base + 8 + j].type._type = nn2[j].type;
  }

  for (int b = 0; b < 8; b++) {
    long int nid = nn1[b].id;
    _type_lattice_id nx = nid % g_meta->size_x;
    _type_lattice_id ny = (nid / g_meta->size_x) % g_meta->size_y;
    _type_lattice_id nz = nid / (g_meta->size_x * g_meta->size_y);

    dev_Lattice t1[8];
    dev_Lattice t2[6];
    get1nn(nx, ny, nz, t1);
    get2nn(nx, ny, nz, t2);

    int off1 = base + 14 + b * 8;
    for (int j = 0; j < 8; j++) {
      out_nn[off1 + j].id = static_cast<int>(t1[j].id);
      out_nn[off1 + j].type._type = t1[j].type;
    }

    int off2 = base + 78 + b * 6;
    for (int j = 0; j < 6; j++) {
      out_nn[off2 + j].id = static_cast<int>(t2[j].id);
      out_nn[off2 + j].type._type = t2[j].type;
    }
  }
}
__global__ void compute_types_for_list(const long int* __restrict__ list,
                                      long int* __restrict__ out_types,
                                      int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    out_types[tid] = getType(list[tid]);
}
__global__ void get_surface_and_ghost_type1(long int *surface_type, long int *ghost_type) {
    _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 处理surface数据
    if (tid < g_surface_Hash->capacity) {
        if (g_surface_Hash->keys[tid] != -1 && g_surface_Hash->keys[tid] != -2) {
            surface_type[tid] = getType(g_surface_Hash->keys[tid]);
        }
    }
    
    // 处理ghost数据（需要调整索引计算）
    _type_lattice_count ghost_tid = tid - g_surface_Hash->capacity;
    if (tid >= g_surface_Hash->capacity && ghost_tid < g_ghost_Hash->capacity) {
        if (g_ghost_Hash->keys[ghost_tid] != -1 && g_ghost_Hash->keys[ghost_tid] != -2) {
            ghost_type[ghost_tid] = getType(g_ghost_Hash->keys[ghost_tid]);
        }
    }
}
void recb_solver_GPU(
                    const unsigned int& sector_id,long int arr_size,
                    long int *&h_ghost_Hash,long int *&h_surface_Hash,
                    long int *&h_ghost_type, long int *&h_surface_type,
                    int *sizes
                  ){
    using namespace gpuSimulate;
    //在这里调用一个单核核函数对hash_set进行更新
    // printf("current arr size is %ld",arr_size);
    // HANDLE_HIP(hipMalloc((void**)&Pair_Atoms,arr_size*sizeof(long int)*2));
    // HANDLE_HIP(hipMemset(Pair_Atoms, -1,arr_size * sizeof(long int) * 2));
   
    // printf("starting recb solver with GPU sector_id:%d,arr_size:%d\n",sector_id,arr_size);
    const bool do_profile = recbProfileEnabled();
    if (do_profile) initRecbProfileEvents();

    blockSize = 128;
    numBlocks = pair_atoms_size / blockSize + 1;
    // print_Hash_and_Pair
    // printf("+++++++++++++++++++++++++++++++++++++++++++++++++\n"); 
    // printf("arr size is %ld\n",pair_atoms_size);
    // hipLaunchKernelGGL(print_Hash_and_Pair, 1, 1, 0, 0,arr_size,0);
  
    if (!d_recb_verify_stats) {
      HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&d_recb_verify_stats), sizeof(RecbVerifyStats)));
    }
    HANDLE_HIP(hipMemset(d_recb_verify_stats, 0, sizeof(RecbVerifyStats)));

    if (do_profile) HANDLE_HIP(hipEventRecord(recb_evt_begin, 0));
    hipLaunchKernelGGL(calcExchangePairs, numBlocks, blockSize, 0, 0,
                      sector_id,pair_atoms_size, d_recb_verify_stats
                      );
    CHECK_FOR_ERROR();
    hipDeviceSynchronize();
    if (do_profile) HANDLE_HIP(hipEventRecord(recb_evt_after_exchange, 0));

    {
      static int recb_launch_by_sector[8] = {};
      const unsigned int sid = sector_id < 8 ? sector_id : 0;
      recb_launch_by_sector[sid]++;
      const int launch_n = recb_launch_by_sector[sid];
      int recb_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &recb_rank);
      if (recb_rank == 0 && launch_n <= 5) {
        RecbVerifyStats h_stats{};
        HANDLE_HIP(hipMemcpy(&h_stats, d_recb_verify_stats, sizeof(RecbVerifyStats), hipMemcpyDeviceToHost));
        kiwi::logs::v(" ", " [recb-verify] sect={} launch={} sector_pairs={} hops={} max_hops_one_pair={} pairs_multi_hop={} pairs_captured_no_move={}.\n",
                      sid, launch_n, h_stats.sector_pairs, h_stats.hops, h_stats.max_hops,
                      h_stats.multi_hop_pairs, h_stats.captured_no_move);
      }
    }
    // print_momo_and_more
    //  hipLaunchKernelGGL(print_momo_and_more, 1, 1, 0, 0
                      
    //                   );
    // hipLaunchKernelGGL(print_surface_and_ghost, 1, 1, 0, 0
    //                   );
    // print_surface_and_ghost

    // printf("---------------------------------------------------\n"); 
    // hipLaunchKernelGGL(print_Hash_and_Pair, 1, 1, 0, 0,arr_size,1);
    // printf("+++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    // hipHostFree(Pair_Atoms);
    

    //这是计算后传回的部分
    //ghost和surface区需要传回

    

    HANDLE_HIP(hipMemset(d_surface_count, 0, sizeof(int)));
    HANDLE_HIP(hipMemset(d_ghost_count, 0, sizeof(int)));
    const int compact_block = 256;
    const int surface_grid = (surface_capacity + compact_block - 1) / compact_block;
    const int ghost_grid = (ghost_capacity + compact_block - 1) / compact_block;
    hipLaunchKernelGGL(compact_valid_keys, surface_grid, compact_block, 0, 0,
                       surface_Hash->device_ptr()->keys, surface_capacity, d_surface_list, d_surface_count);
    CHECK_FOR_ERROR();
    hipLaunchKernelGGL(compact_valid_keys, ghost_grid, compact_block, 0, 0,
                       ghost_Hash->device_ptr()->keys, ghost_capacity, d_ghost_list, d_ghost_count);
    CHECK_FOR_ERROR();
    hipDeviceSynchronize();

    int h_surface_count = 0;
    int h_ghost_count = 0;
    HANDLE_HIP(hipMemcpy(&h_surface_count, d_surface_count, sizeof(int), hipMemcpyDeviceToHost));
    HANDLE_HIP(hipMemcpy(&h_ghost_count, d_ghost_count, sizeof(int), hipMemcpyDeviceToHost));

    commClearGhostBuckets();
    if (h_ghost_count > 0) {
      commBuildGhostBucketsFromList(sector_id, d_ghost_list, h_ghost_count);
    }
    commClearSurfaceBuckets();
    if (h_surface_count > 0) {
      commBuildSurfaceBucketsFromList(d_surface_list, h_surface_count);
    }

    const int type_block = 256;
    if (h_surface_count > 0) {
      const int grid = (h_surface_count + type_block - 1) / type_block;
      hipLaunchKernelGGL(compute_types_for_list, grid, type_block, 0, 0,
                         d_surface_list, d_surface_list_type, h_surface_count);
      CHECK_FOR_ERROR();
    }
    if (h_ghost_count > 0) {
      const int grid = (h_ghost_count + type_block - 1) / type_block;
      hipLaunchKernelGGL(compute_types_for_list, grid, type_block, 0, 0,
                         d_ghost_list, d_ghost_list_type, h_ghost_count);
      CHECK_FOR_ERROR();
    }
    hipDeviceSynchronize();
    if (do_profile) HANDLE_HIP(hipEventRecord(recb_evt_after_type, 0));

    if (h_ghost_count > 0 && h_ghost_Hash && h_ghost_type) {
      HANDLE_HIP(hipMemcpy(h_ghost_Hash, d_ghost_list, h_ghost_count * sizeof(long int), hipMemcpyDeviceToHost));
      HANDLE_HIP(hipMemcpy(h_ghost_type, d_ghost_list_type, h_ghost_count * sizeof(long int), hipMemcpyDeviceToHost));
    }
    if (h_surface_count > 0 && h_surface_Hash && h_surface_type) {
      HANDLE_HIP(hipMemcpy(h_surface_Hash, d_surface_list, h_surface_count * sizeof(long int), hipMemcpyDeviceToHost));
      HANDLE_HIP(hipMemcpy(h_surface_type, d_surface_list_type, h_surface_count * sizeof(long int), hipMemcpyDeviceToHost));
    }
    if (do_profile) HANDLE_HIP(hipEventRecord(recb_evt_after_copy, 0));

    ghost_Hash->clear();
    surface_Hash->clear();
    if (do_profile) HANDLE_HIP(hipEventRecord(recb_evt_after_clear, 0));
    // for(int i=0;i<surface_Hash->device_ptr()->capacity;i++){
    //   printf("%ld-%ld\t",h_surface_Hash[i],h_surface_type[i]);
    //   if(i%20 == 0)
    //   printf("\n");
    // }
    if (sizes) {
      sizes[0]= h_ghost_count;
      sizes[1]= h_surface_count;
    }

    if (do_profile) {
      HANDLE_HIP(hipEventSynchronize(recb_evt_after_clear));
      HANDLE_HIP(hipEventElapsedTime(&recb_ms_exchange, recb_evt_begin, recb_evt_after_exchange));
      HANDLE_HIP(hipEventElapsedTime(&recb_ms_type, recb_evt_after_exchange, recb_evt_after_type));
      HANDLE_HIP(hipEventElapsedTime(&recb_ms_copy, recb_evt_after_type, recb_evt_after_copy));
      HANDLE_HIP(hipEventElapsedTime(&recb_ms_clear, recb_evt_after_copy, recb_evt_after_clear));
      // std::cout
      //     << "[KMC_RECB_PROFILE] N=" << pair_atoms_size
      //     << " exchange_ms=" << recb_ms_exchange
      //     << " type_ms=" << recb_ms_type
      //     << " copy_ms=" << recb_ms_copy
      //     << " clear_ms=" << recb_ms_clear
      //     << std::endl;
    }
    // printf("cal exchange finished \n");
}

double calculate_GPU(_type_lattice_id *vac_idArray,  const _type_lattice_count& vac_count) {
  using namespace gpuSimulate;
  
  *sum_rates = 0;
  currentVacCount = vac_count;
  // 如果超出预分配数组大小则重新分配锁页内存和显存
  if(vac_count > vacHostArrayCapacity) {
    vacHostArrayCapacity = vac_count * RE_SCALE_SIZE;
    hipHostFree(hostVacancyInfo);
    hipHostFree(hostNeighbourInfo);
    HANDLE_HIP(hipFree(deviceVacancyInfo));
    HANDLE_HIP(hipFree(deviceNeighbourInfo));

    hipHostMalloc((void**)&hostVacancyInfo, vacHostArrayCapacity * sizeof(dev_Vacancy));
    hipHostMalloc((void**)&hostNeighbourInfo, vacHostArrayCapacity * NN_TOTAL * sizeof(dev_nnLattice));
    HANDLE_HIP(hipMalloc((void**)&deviceVacancyInfo, vacHostArrayCapacity * sizeof(dev_Vacancy)));
    HANDLE_HIP(hipMalloc((void**)&deviceNeighbourInfo, vacHostArrayCapacity * NN_TOTAL * sizeof(dev_nnLattice)));
  }

  lattice_list->initGpuInfo(vac_count, vac_idArray, hostVacancyInfo, hostNeighbourInfo);//从哈希表找数据传输到gpu
  HANDLE_HIP(hipMemcpy(deviceVacancyInfo, hostVacancyInfo, vac_count * sizeof(dev_Vacancy), hipMemcpyHostToDevice));
  HANDLE_HIP(hipMemcpy(deviceNeighbourInfo, hostNeighbourInfo, vac_count * NN_TOTAL * sizeof(dev_nnLattice), hipMemcpyHostToDevice));
  // for (_type_lattice_count j = 0; j < vac_count; j++) {
  //   kiwi::logs::v(" ", " latti_id is : {}. \n", hostVacancyInfo[j].id);
  //   for (int i = 0; i < 8; i++) {
  //     kiwi::logs::v(" ", " nn_index is : {} nn_id is : {} nn_type is : {} . \n", i, deviceNeighbourInfo[j * NN_TOTAL + i].id, deviceNeighbourInfo[j * NN_TOTAL + i].type._type);
  //   }
  // }
  blockSize = 128;
  numBlocks = (vac_count + blockSize - 1) / blockSize;

  // 调用GPU内核函数
  hipLaunchKernelGGL(calcRatesGPU, numBlocks, blockSize, 0, 0, 
                    deviceVacancyInfo,
                    deviceNeighbourInfo,
                    d_sum_rates,
                    h_config[0],
                    h_config[1], 
                    vac_count);

  free(vac_idArray);

  CHECK_FOR_ERROR();
  HANDLE_HIP(hipMemcpy(sum_rates, d_sum_rates, sizeof(double), hipMemcpyDeviceToHost));
  return *sum_rates;
}

double calculate_GPU_region(int x_low, int x_high, int y_low, int y_high, int z_low, int z_high) {
  using namespace gpuSimulate;
  HANDLE_HIP(hipMemset(d_sum_rates, 0, sizeof(double)));
  *sum_rates = 0;

  HANDLE_HIP(hipMemset(d_vac_count_all, 0, sizeof(int)));
  const int compact_block = 256;
  const int vac_grid = (vac_capacity_all + compact_block - 1) / compact_block;
  hipLaunchKernelGGL(compact_valid_keys, vac_grid, compact_block, 0, 0,
                     V_Hash->device_ptr()->keys, vac_capacity_all, d_vac_all, d_vac_count_all);
  CHECK_FOR_ERROR();
  hipDeviceSynchronize();

  int h_vac_all = 0;
  HANDLE_HIP(hipMemcpy(&h_vac_all, d_vac_count_all, sizeof(int), hipMemcpyDeviceToHost));
  if (h_vac_all <= 0) return 0.0;

  HANDLE_HIP(hipMemset(d_vac_count_sector, 0, sizeof(int)));
  const int filter_block = 256;
  const int filter_grid = (h_vac_all + filter_block - 1) / filter_block;
  hipLaunchKernelGGL(filter_ids_in_region, filter_grid, filter_block, 0, 0,
                     d_vac_all, h_vac_all,
                     x_low, x_high, y_low, y_high, z_low, z_high,
                     d_vac_sector, d_vac_count_sector);
  CHECK_FOR_ERROR();
  hipDeviceSynchronize();

  int h_vac_sector = 0;
  HANDLE_HIP(hipMemcpy(&h_vac_sector, d_vac_count_sector, sizeof(int), hipMemcpyDeviceToHost));
  if (h_vac_sector <= 0) return 0.0;

  currentVacCount = h_vac_sector;
  if (static_cast<_type_lattice_count>(h_vac_sector) > vacHostArrayCapacity) {
    vacHostArrayCapacity = static_cast<_type_lattice_count>(h_vac_sector);
    HANDLE_HIP(hipFree(deviceVacancyInfo));
    HANDLE_HIP(hipFree(deviceNeighbourInfo));
    HANDLE_HIP(hipMalloc((void**)&deviceVacancyInfo, vacHostArrayCapacity * sizeof(dev_Vacancy)));
    HANDLE_HIP(hipMalloc((void**)&deviceNeighbourInfo, vacHostArrayCapacity * NN_TOTAL * sizeof(dev_nnLattice)));
  }

  blockSize = 128;
  numBlocks = (h_vac_sector + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(build_nn_info_from_vac_list, numBlocks, blockSize, 0, 0,
                     d_vac_sector, h_vac_sector, deviceVacancyInfo, deviceNeighbourInfo);
  CHECK_FOR_ERROR();

  hipLaunchKernelGGL(calcRatesGPU, numBlocks, blockSize, 0, 0,
                     deviceVacancyInfo,
                     deviceNeighbourInfo,
                     d_sum_rates,
                     h_config[0],
                     h_config[1],
                     h_vac_sector);
  CHECK_FOR_ERROR();
  HANDLE_HIP(hipMemcpy(sum_rates, d_sum_rates, sizeof(double), hipMemcpyDeviceToHost));
  return *sum_rates;
}
// void initialize_gpu(int process_rank){
//   int gpu_count;
//   HANDLE_HIP(hipGetDeviceCount(&gpu_count));

//   int dev = process_rank % gpu_count;
//   std::cerr << "[initialize_gpu] rank " << process_rank
//             << " gpu_count=" << gpu_count
//             << " before hipSetDevice dev=" << dev << std::endl;

//   HANDLE_HIP(hipSetDevice(dev));

//   int current_dev = -1;
//   HANDLE_HIP(hipGetDevice(&current_dev));
//   std::cerr << "[initialize_gpu] rank " << process_rank
//             << " after hipSetDevice current_dev=" << current_dev << std::endl;

//   std::cerr << "[initialize_gpu] rank " << process_rank
//             << " before hipDeviceReset" << std::endl;
//   HANDLE_HIP(hipDeviceReset());

//   HANDLE_HIP(hipGetDevice(&current_dev));
//   std::cerr << "[initialize_gpu] rank " << process_rank
//             << " after hipDeviceReset current_dev=" << current_dev << std::endl;

// #ifdef KMC_RCCL_ENABLE
//   std::cerr << "[initialize_gpu] rank " << process_rank
//             << " before rccl_smoke_test" << std::endl;
//   rccl_smoke_test();
//   std::cerr << "[initialize_gpu] rank " << process_rank
//             << " after rccl_smoke_test" << std::endl;
// #endif
// }


void initialize_gpu(int process_rank){
  int gpu_count;
  HANDLE_HIP(hipGetDeviceCount(&gpu_count));
  const int local_rank = getLocalProcessRank(process_rank);
  const int device_id = local_rank % gpu_count;
  HANDLE_HIP(hipSetDevice(device_id));
  if (shouldPrintGpuBindDebug(process_rank, local_rank)) {
    std::cerr << "[KMC_GPU_BIND] process_rank=" << process_rank
              << " local_rank=" << local_rank
              << " gpu_count=" << gpu_count
              << " device_id=" << device_id << std::endl;
  }
  HANDLE_HIP(hipDeviceReset());
#ifdef KMC_RCCL_ENABLE
  rccl_smoke_test();
#endif
}

void gpu_prepare(double& v, double& T, LatticesList *p_list, comm::ColoredDomain *_p_domain,
                std::vector<long int> pair_atoms
                ){
  using namespace gpuSimulate;

  int device_id;
  hipGetDevice(&device_id);  // 获取当前设备ID
// size_t free_mem = 0;
// size_t total_mem = 0;
// HANDLE_HIP(hipMemGetInfo(&free_mem, &total_mem));

// std::cerr << "[gpu_prepare] current device=" << device_id
//           << " free_mem=" << free_mem
//           << " total_mem=" << total_mem
//           << std::endl;

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, device_id);  // 获取设备属性
  std::cout << "GPU名称: " << prop.name << std::endl;
  std::cout << "每个线程块最大线程数: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "每个多处理器最大线程块数: " << prop.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "线程块最大尺寸（x,y,z）: (" 
            << prop.maxThreadsDim[0] << ", " 
            << prop.maxThreadsDim[1] << ", " 
            << prop.maxThreadsDim[2] << ")" << std::endl;
  std::cout << "网格最大尺寸（x,y,z）: (" 
            << prop.maxGridSize[0] << ", " 
            << prop.maxGridSize[1] << ", " 
            << prop.maxGridSize[2] << ")" << std::endl;

  lattice_list = p_list;
  p_domain = _p_domain;
  h_config[0] = v;   // v
  h_config[1] = T;   // T
  std::vector<long int> region_msg(8 * 6);

  
  int index =0;
  for(auto region_sector:p_domain->local_sector_region){
    region_msg[index * 6 + 0] = region_sector.x_low;
    region_msg[index * 6 + 1] = region_sector.x_high;
    region_msg[index * 6 + 2] = region_sector.y_low;
    region_msg[index * 6 + 3] = region_sector.y_high;
    region_msg[index * 6 + 4] = region_sector.z_low;
    region_msg[index * 6 + 5] = region_sector.z_high;
    index++;
  }
  index = 0;
  printf("init region msg \n");
  HANDLE_HIP(hipMalloc((void**)&d_sum_rates, sizeof(double)));
  HANDLE_HIP(hipMalloc((void**)&d_event, sizeof(dev_event)));
  
  vacHostArrayCapacity = lattice_list->vac_hash.size() + 1;
  HostAtomCapacity = lattice_list->momo_hash.size()
                      + lattice_list->more_hash.size()
                      +lattice_list->rere_hash.size();

  HANDLE_HIP(hipMalloc((void**)&deviceVacancyInfo, vacHostArrayCapacity * sizeof(dev_Vacancy)));
  HANDLE_HIP(hipMalloc((void**)&deviceNeighbourInfo, vacHostArrayCapacity * NN_TOTAL * sizeof(dev_nnLattice)));

  hipHostMalloc((void**)&sum_rates, sizeof(double));
  hipHostMalloc((void**)&h_event, sizeof(dev_event));

  hipHostMalloc((void**)&hostVacancyInfo, vacHostArrayCapacity * sizeof(dev_Vacancy));
  hipHostMalloc((void**)&hostNeighbourInfo, vacHostArrayCapacity * NN_TOTAL * sizeof(dev_nnLattice));


  hipHostMalloc((void**)&atoms, HostAtomCapacity * sizeof(dev_atom));
  hipHostMalloc((void**)&ret_atoms, HostAtomCapacity * sizeof(dev_atom));
  HANDLE_HIP(hipMalloc((void**)&d_atoms, HostAtomCapacity * sizeof(dev_atom)));

  //以下是对recb_solver_GPU的空间分配
  //在这里直接把GPU中的哈希表分配掉了，然后在空位交换的时候更新哈希表
  /*
    变量名           位置         解释
    Pair_Atoms      host      间隙对的列表
    d_Pair_Atoms    device    间隙对的列表（GPU）
    meta            host      box中环境变量的集合
    d_meta          device    box中环境变量的集合
    ghost_size      host      ghost区的大小（现在按最大的情况来）
    surface_size    host      surface区的大小（现在按最大的情况来）
    MoRe_Hash       device    GPU中哈希表的位置
    h_MoRe_Hash     host      计算完毕后传回给CPU的空间
    MoMo_Hash等等同理
    busy_set        host      忙碌队列
  */
  pair_atoms_size = pair_atoms.size()*2;
  hipHostMalloc((void**)&Pair_Atoms,pair_atoms.size()*sizeof(long int));
  HANDLE_HIP(hipMalloc((void**)&d_Pair_Atoms,pair_atoms.size()*sizeof(long int)*2));
  HANDLE_HIP(hipMemset(d_Pair_Atoms, -1, pair_atoms.size() * sizeof(long int) * 2));
  //对环境变量以及哈希表进行初始化
  //每组的meta需要重新赋值
  // printf("init gpu_info msg \n");
  lattice_list->initGpuInfo_exchange(pair_atoms,region_msg,
                                    0,meta,Pair_Atoms,
                                    MoRe_Hash,MoMo_Hash,Re_Hash,V_Hash,Busy_Set
                                    );
  // printf("init gpu_info msg end\n");
  HANDLE_HIP(hipMemcpy(d_Pair_Atoms, Pair_Atoms,pair_atoms.size()*sizeof(long int), hipMemcpyHostToDevice));

  // vac_capacity_all = V_Hash->capacity();
  vac_capacity_all = V_Hash->get_capacity();
  vac_capacity_sector = vac_capacity_all;
  HANDLE_HIP(hipMalloc((void**)&d_vac_all, vac_capacity_all * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_vac_sector, vac_capacity_sector * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_vac_count_all, sizeof(int)));
  HANDLE_HIP(hipMalloc((void**)&d_vac_count_sector, sizeof(int)));

  int ghost_size = (meta.regions[1] + meta.ghost_x - meta.regions[0])*(meta.regions[3] + meta.ghost_y - meta.regions[2]) * (meta.regions[5] + meta.ghost_z - meta.regions[4])
                    - (meta.regions[1] - meta.regions[0])*(meta.regions[3] - meta.regions[2]) * (meta.regions[5] - meta.regions[4])
                    + (meta.regions[1]-meta.regions[0])*meta.ghost_y*meta.ghost_z
                    + meta.ghost_x*(meta.regions[3]-meta.regions[2])*meta.ghost_z
                    + meta.ghost_x*(meta.regions[5]-meta.regions[4])*meta.ghost_y;
  int surface_size = (meta.regions[1] - meta.regions[0])*(meta.regions[3] - meta.regions[2]) * (meta.regions[5] - meta.regions[4])
                      - (meta.regions[1] - meta.ghost_x - meta.regions[0])*(meta.regions[3] - meta.ghost_y - meta.regions[2]) * (meta.regions[5] - meta.ghost_z - meta.regions[4]);

  surface_Hash = new HIPHashSet(surface_size);
  ghost_Hash = new HIPHashSet(ghost_size);
  // surface_capacity = surface_size * 2;
  // ghost_capacity = ghost_size * 2;
  surface_capacity = surface_Hash->get_capacity(); 
  ghost_capacity = ghost_Hash->get_capacity();
  commInitGhostBuckets(ghost_size);
  commInitSurfaceBuckets(surface_size);
  {
    const int dims[comm::DIMENSION_SIZE] = {comm::DIM_X, comm::DIM_Y, comm::DIM_Z};
    std::vector<CommBox> boxes;
    std::array<int, 25> offsets{};
    offsets[0] = 0;
    int bucket = 0;
    for (int sect = 0; sect < 8; sect++) {
      for (int di = 0; di < comm::DIMENSION_SIZE; di++) {
        auto regs = comm::fwCommSectorSendRegion(sect, dims[di], p_domain->lattice_size_ghost,
                                                 p_domain->local_split_coord, p_domain->local_sub_box_lattice_region);
        for (auto &r : regs) {
          CommBox b;
          b.x_low = 2 * r.x_low;
          b.x_high = 2 * r.x_high;
          b.y_low = r.y_low;
          b.y_high = r.y_high;
          b.z_low = r.z_low;
          b.z_high = r.z_high;
          boxes.push_back(b);
        }
        offsets[bucket + 1] = static_cast<int>(boxes.size());
        bucket++;
      }
    }
    commSetSurfaceBoxes(boxes.data(), static_cast<int>(boxes.size()), offsets.data(), static_cast<int>(offsets.size()));
  }
  HANDLE_HIP(hipMalloc((void**)&d_surface_list, surface_capacity * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_ghost_list, ghost_capacity * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_surface_list_type, surface_capacity * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_ghost_list_type, ghost_capacity * sizeof(long int)));
  HANDLE_HIP(hipMalloc((void**)&d_surface_count, sizeof(int)));
  HANDLE_HIP(hipMalloc((void**)&d_ghost_count, sizeof(int)));
  //这是对meta的首次更新，以后每次都得更新
  HANDLE_HIP(hipMalloc((void**)&d_meta,sizeof(dev_meta)));
  HANDLE_HIP(hipMemcpy(d_meta, &meta,sizeof(dev_meta), hipMemcpyHostToDevice));
  // printf("init global pointer\n");
  //set hash pointer to gpu global pointer
  // printDevMeta(meta);
//=======分配全局物理位图锁=======
  long int total_lattice_sites = meta.size_x * meta.size_y * meta.size_z; 
  long int num_ull = (total_lattice_sites + 63) / 64; 
  bitmap_lock_bytes = num_ull * sizeof(unsigned long long int);

  HANDLE_HIP(hipMalloc((void**)&d_Global_Bitmap_Lock, bitmap_lock_bytes));
  HANDLE_HIP(hipMemset(d_Global_Bitmap_Lock, 0, bitmap_lock_bytes));
  // HANDLE_HIP(hipMalloc((void**)&d_Global_Bitmap_Lock, num_ull * sizeof(unsigned long long int)));
  // HANDLE_HIP(hipMemset(d_Global_Bitmap_Lock, 0, num_ull * sizeof(unsigned long long int)));
  //===============================
  arr_size = pair_atoms.size();
  // printf("arr_size is %ld",arr_size);
  hipLaunchKernelGGL(setGlobalPointer, 1, 1, 0, 0, 
      d_meta,
      MoRe_Hash->device_ptr(),MoMo_Hash->device_ptr(),Re_Hash->device_ptr(),V_Hash->device_ptr(),Busy_Set->device_ptr(),
      surface_Hash->device_ptr(),ghost_Hash->device_ptr(),d_Pair_Atoms,d_Global_Bitmap_Lock
  );

  hipDeviceSynchronize();
  // std::cin.get();
  // GPUHashSet* hostPtr;
  // hipMemcpyFromSymbol(&hostPtr, g_MoMo_Hash, sizeof(GPUHashSet*));
  // printf("Global pointer value after synchronization: %p\n", hostPtr);
  // hipLaunchKernelGGL(printpointer, 1, 1, 0, 0,   );
  // system("read -p 'Press Enter to continue...' var");


  //这些是对返回CPU的空间
    // printf("malloc CPU hashset\n");
    // hipHostMalloc((void**)&lattice_list->h_MoRe_Hash,MoRe_Hash->device_ptr()->capacity*sizeof(long int));
    // hipHostMalloc((void**)&lattice_list->h_MoMo_Hash,MoMo_Hash->device_ptr()->capacity*sizeof(long int));
    // hipHostMalloc((void**)&lattice_list->h_Re_Hash,Re_Hash->device_ptr()->capacity*sizeof(long int));
    // hipHostMalloc((void**)&lattice_list->h_V_Hash,V_Hash->device_ptr()->capacity*sizeof(long int));

    hipHostMalloc((void**)&lattice_list->h_ghost_Hash, ghost_capacity * sizeof(long int));
    hipHostMalloc((void**)&lattice_list->h_surface_Hash, surface_capacity * sizeof(long int));

    hipHostMalloc((void**)&lattice_list->h_ghost_Hash_type, ghost_capacity * sizeof(long int));
    hipHostMalloc((void**)&lattice_list->h_surface_Hash_type, surface_capacity * sizeof(long int));

  // printf("finish gpu prepare\n");
  hipHostFree( Pair_Atoms);
//   HANDLE_HIP(hipDeviceSynchronize());

// size_t free_mem_end = 0;
// size_t total_mem_end = 0;
// HANDLE_HIP(hipMemGetInfo(&free_mem_end, &total_mem_end));

// std::cerr << "[gpu_prepare] end current device=" << device_id
//           << " free_mem=" << free_mem_end
//           << " total_mem=" << total_mem_end
//           << std::endl;
}

void gpu_final(){
  using namespace gpuSimulate;

  destroyRecbProfileEvents();

  // 释放GPU内存
  HANDLE_HIP(hipFree(d_sum_rates));
  HANDLE_HIP(hipFree(d_event));

  HANDLE_HIP(hipFree(deviceVacancyInfo));
  HANDLE_HIP(hipFree(deviceNeighbourInfo));
  HANDLE_HIP(hipFree(d_vac_all));
  HANDLE_HIP(hipFree(d_vac_sector));
  HANDLE_HIP(hipFree(d_vac_count_all));
  HANDLE_HIP(hipFree(d_vac_count_sector));

  hipHostFree(h_event);
  hipHostFree(sum_rates);

  hipHostFree(hostVacancyInfo);
  hipHostFree(hostNeighbourInfo);

if (d_Global_Bitmap_Lock != nullptr) {
    HANDLE_HIP(hipFree(d_Global_Bitmap_Lock));
  }
  //z这里是recb_solver_GPU的部分
  delete MoRe_Hash;
  delete MoMo_Hash;
  delete Re_Hash;
  delete V_Hash;

  HANDLE_HIP(hipFree(d_surface_list));
  HANDLE_HIP(hipFree(d_ghost_list));
  HANDLE_HIP(hipFree(d_surface_list_type));
  HANDLE_HIP(hipFree(d_ghost_list_type));
  HANDLE_HIP(hipFree(d_surface_count));
  HANDLE_HIP(hipFree(d_ghost_count));

  hipFree(d_Pair_Atoms);
  if (d_recb_verify_stats) {
    HANDLE_HIP(hipFree(d_recb_verify_stats));
    d_recb_verify_stats = nullptr;
  }


  // hipHostFree(Pair_Atoms);

  delete ghost_Hash;
  delete surface_Hash;
  delete comm_ghost_bucket0;
  delete comm_ghost_bucket1;
  delete comm_ghost_bucket2;
  delete comm_ghost_bucket3;
  delete comm_ghost_bucket4;
  delete comm_ghost_bucket5;
  delete comm_ghost_bucket6;
  if (d_comm_ghost_regions) HANDLE_HIP(hipFree(d_comm_ghost_regions));
  for (int i = 0; i < 24; i++) {
    delete comm_surface_buckets[i];
  }
  if (d_comm_surface_bucket_ptrs) HANDLE_HIP(hipFree(d_comm_surface_bucket_ptrs));
  if (d_comm_surface_boxes) HANDLE_HIP(hipFree(d_comm_surface_boxes));
  if (d_comm_surface_offsets) HANDLE_HIP(hipFree(d_comm_surface_offsets));

  delete g_Busy_Set;
  hipFree(d_meta);
}

void exchange_lattices(dev_event* h_event){
  using namespace gpuSimulate;

  if (lattice_list->vac_hash.count(h_event->from_id)) {
    lattice_list->vac_hash.erase(h_event->from_id);
    lattice_list->vac_hash.emplace(std::make_pair(h_event->to_id, VacancyHash{}));
  } else assert(0);

  switch (h_event->to_type._type)
  {
    case LatticeTypes::Re:
      lattice_list->re_hash.erase(h_event->to_id);
      lattice_list->re_hash.emplace(h_event->from_id);
      break;
    case LatticeTypes::Mn:
      lattice_list->mn_hash.erase(h_event->to_id);
      lattice_list->mn_hash.emplace(h_event->from_id);
      break;
    case LatticeTypes::Ni:
      lattice_list->ni_hash.erase(h_event->to_id);
      lattice_list->ni_hash.emplace(h_event->from_id);
      break;
    case LatticeTypes::Si:
      lattice_list->si_hash.erase(h_event->to_id);
      lattice_list->si_hash.emplace(h_event->from_id);
      break;
    case LatticeTypes::Mo:
      break;
    default:
      assert(0);
      break;
  }
}

void add_surface(_type_lattice_id& surface_id,
                std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_x, 
                std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_y,
                std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_z) {
  using namespace gpuSimulate;

  _type_lattice_id surface_x = surface_id % lattice_list->meta.size_x;
  _type_lattice_id surface_y = (surface_id / lattice_list->meta.size_x) % lattice_list->meta.size_y;
  _type_lattice_id surface_z = surface_id / (lattice_list->meta.size_x * lattice_list->meta.size_y);
  const int dims[comm::DIMENSION_SIZE] = {comm::DIM_X, comm::DIM_Y, comm::DIM_Z};
  // todo the regions can be static.
  std::array<std::vector<comm::Region<comm::_type_lattice_coord>>, comm::DIMENSION_SIZE> send_regions; // send regions in each dimension
  // 应该可以不循环 8 次的 ？ 如果发送区没有重叠的话
  for (int sect = 0; sect < 8; sect++) {
    for (int d = 0; d < comm::DIMENSION_SIZE; d++) {
      send_regions[d] = comm::fwCommSectorSendRegion(sect, dims[d], p_domain->lattice_size_ghost,
                                                    p_domain->local_split_coord, p_domain->local_sub_box_lattice_region);
    }
    for(int d = 0; d < comm::DIMENSION_SIZE; d++) {
      for (auto &r : send_regions[d]) {
        _type_lattice_size x_low = 2 * r.x_low;
        _type_lattice_size y_low = r.y_low;
        _type_lattice_size z_low = r.z_low;
        _type_lattice_size x_high = 2 * r.x_high;
        _type_lattice_size y_high = r.y_high;
        _type_lattice_size z_high = r.z_high;
    
        if(x_low <= surface_x && surface_x < x_high && y_low <= surface_y && surface_y < y_high && z_low <= surface_z && surface_z < z_high) {
          if(d == 0){
            if(!exchange_surface_x[sect].count(surface_id)) exchange_surface_x[sect].emplace(surface_id);
          } else if(d == 1) {
            if(!exchange_surface_y[sect].count(surface_id)) exchange_surface_y[sect].emplace(surface_id);
          } else {
            if(!exchange_surface_z[sect].count(surface_id)) exchange_surface_z[sect].emplace(surface_id);
          }
        }
      }
    }
  }
}

void output_time() {
  using namespace gpuSimulate;
  // kiwi::logs::v(" ", " time_event is : {} s. \n", time_event);
  // kiwi::logs::v(" ", " time_event_trans is : {} s. \n", time_event_trans);
  // kiwi::logs::v(" ", " time_event_other : {} s. \n", time_event_other);
  // kiwi::logs::v(" ", " time_update_construct : {} s. \n", time_update_construct);
  // kiwi::logs::v(" ", " time_update_trans : {} s. \n", time_update_trans);
}
__global__ void update_hash_set(ChangeLattice_GPU *d_buffer,int receive_len,const int dimension,
                                int sector_id,
                                long int sub_box_lattice_size_0,long int sub_box_lattice_size_1,long int sub_box_lattice_size_2,
                                long int  neighbour_local_sub_box_0,long int neighbour_local_sub_box_1,long int neighbour_local_sub_box_2
){
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<KMC_HASH_TILE_SIZE>(block);

    long int BCC_DBX = 2;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 初始化有效性掩码==注意不要return
    bool active = (index < receive_len);

    long int x=0,y=0,z=0;
    long int o_x=0,o_y=0,o_z=0;
    long int id=0,o_id=0;
   int type = 0;

    if(active){
      o_x= d_buffer[index].x;
      o_y= d_buffer[index].y;
      o_z= d_buffer[index].z;
      o_id = getId(d_buffer[index].x,d_buffer[index].y,d_buffer[index].z);

      type = d_buffer[index].type;
      if(dimension == 0){
        if(sector_id % 2 == 0){
          d_buffer[index].x += BCC_DBX * sub_box_lattice_size_0;
        }else{
          d_buffer[index].x -= BCC_DBX * (sub_box_lattice_size_0 + (neighbour_local_sub_box_0 - sub_box_lattice_size_0));
        }
        x = d_buffer[index].x;
        y = d_buffer[index].y;
        z = d_buffer[index].z;
        id = getId(x,y,z);
        // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	,change sur pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
      }else if(dimension == 1){
    
        x = d_buffer[index].x;
        if(sector_id == 0 || sector_id == 1 || sector_id == 4 || sector_id == 5) {
          d_buffer[index].y += sub_box_lattice_size_1;
        } else {
          d_buffer[index].y -= sub_box_lattice_size_1 + (neighbour_local_sub_box_1 - sub_box_lattice_size_1);
        }
        y = d_buffer[index].y;
        z = d_buffer[index].z;

        id = getId(x,y,z);
        // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	,change sur pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
      
      }else if(dimension == 2){
        x = d_buffer[index].x;
        y = d_buffer[index].y;
        if(sector_id < 4) {
          d_buffer[index].z += sub_box_lattice_size_2;
        }else{
          d_buffer[index].z -= sub_box_lattice_size_2+ (neighbour_local_sub_box_2 - sub_box_lattice_size_2);
        }
        z = d_buffer[index].z;

        id = getId(x,y,z);
        // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	,change sur pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
      }
    }
    // L1: 拦截 Buffer 中传来的重复 ID
    resolve_intra_tile_conflicts_single(id, active, tile);

    // L2: 拦截跨 Block 的并发修改
    bool lock_acquired = false;
    if (active) {
        lock_acquired = try_acquire_bitmap_lock_single(id);
        if (!lock_acquired) active = false; // 抢锁失败，本轮不操作
    }
    //
    bool need_del = active;
    // 级联删除， 物理优化：按 Mo-Re 合金的浓度与反应产物留存概率，从高到低排序
    // 既然在 V 删了，就不用去查 Re, MoMo 了，直接把 active_remove 设为 false。
    // 这样接下来的 remove 操作就会直接被 work_mask 拦截，耗时为 0。
    if (tiled_remove_op(g_Re_Hash, id, tile, need_del,g_Re_Hash->num_buckets) == 1) {
        need_del = false; 
    }
    if (tiled_remove_op(g_MoRe_Hash, id, tile, need_del,g_MoRe_Hash->num_buckets) == 1) {
        need_del = false; 
    }
    if (tiled_remove_op(g_MoMo_Hash, id, tile, need_del,g_MoMo_Hash->num_buckets) == 1) {
        need_del = false; 
    }
    tiled_remove_op(g_V_Hash, id, tile, need_del,g_V_Hash->num_buckets);

    bool insert_V    = (active && type == LatticeTypes::V);
    bool insert_Re   = (active && type == LatticeTypes::Re);
    bool insert_MoMo = (active && type == LatticeTypes::MoMo);
    bool insert_MoRe = (active && type == LatticeTypes::MoRe);

    tiled_insert_op(g_V_Hash, id, tile, insert_V,g_V_Hash->num_buckets);
    tiled_insert_op(g_Re_Hash, id, tile, insert_Re,g_Re_Hash->num_buckets);
    tiled_insert_op(g_MoMo_Hash, id, tile, insert_MoMo,g_MoMo_Hash->num_buckets);
    tiled_insert_op(g_MoRe_Hash, id, tile, insert_MoRe,g_MoRe_Hash->num_buckets);
    if (lock_acquired) {
        release_bitmap_lock_single(id);
    }
    return;
}


__global__ void update_hash_set_tiled(ChangeLattice_GPU *d_buffer,int receive_len,const int dimension,
                                int sector_id,
                                long int sub_box_lattice_size_0,long int sub_box_lattice_size_1,long int sub_box_lattice_size_2,
                                long int  neighbour_local_sub_box_0,long int neighbour_local_sub_box_1,long int neighbour_local_sub_box_2
){
  // printf("env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
  //       neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2);

  long int BCC_DBX = 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= receive_len - 1)
    return;

  long int  x,y,z=0;
  long int o_x=0,o_y=0,o_z=0;
  long int id=0,o_id=0;
  o_x= d_buffer[index].x;
  o_y= d_buffer[index].y;
  o_z= d_buffer[index].y;
  o_id = getId(d_buffer[index].x,d_buffer[index].y,d_buffer[index].z);
  if(dimension == 0){
    // for(int i=index;i<receive_len;i+=blockDim.x)
    {
      // printf("before change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("cal env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld------%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2,BCC_DBX);
      // printf("",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      if(sector_id % 2 == 0){
        d_buffer[index].x += BCC_DBX * sub_box_lattice_size_0;
      }else{
        d_buffer[index].x -= BCC_DBX * (sub_box_lattice_size_0 + (neighbour_local_sub_box_0 - sub_box_lattice_size_0));
      }
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      z = d_buffer[index].z;
      id = getId(x,y,z);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	,change sur pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove_atom_op(g_V_Hash,id);
        // printf("--V--");
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove_atom_op(g_Re_Hash,id);
        //  printf("--Re--");
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove_atom_op(g_MoMo_Hash,id);
        //  printf("--MoMo--");
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove_atom_op(g_MoRe_Hash,id);
        //  printf("--MoRe--");
      }
      // printf("%ld\n",d_buffer[index].type);
      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert_atom_op(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
           hash_set_insert_atom_op(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert_atom_op(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert_atom_op(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }

  }
  else if(dimension == 1){
    // for(int i=index;i<receive_len;i+=blockDim.x)
    {
      // printf("before change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("cal env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld------%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2,BCC_DBX);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      x = d_buffer[index].x;
      if(sector_id == 0 || sector_id == 1 || sector_id == 4 || sector_id == 5) {
        d_buffer[index].y += sub_box_lattice_size_1;
      } else {
        d_buffer[index].y -= sub_box_lattice_size_1 + (neighbour_local_sub_box_1 - sub_box_lattice_size_1);
      }
      y = d_buffer[index].y;
      z = d_buffer[index].z;

      id = getId(x,y,z);
       //printf("change pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);

      // printf("change sur pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	,change sur pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove_atom_op(g_V_Hash,id);
        // printf("--V--");
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove_atom_op(g_Re_Hash,id);
        //  printf("--Re--");
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove_atom_op(g_MoMo_Hash,id);
        //  printf("--MoMo--");
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove_atom_op(g_MoRe_Hash,id);
        //  printf("--MoRe--");
      }
      // printf("%ld\n",d_buffer[index].type);

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert_atom_op(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert_atom_op(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert_atom_op(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert_atom_op(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }
  }
  else if(dimension == 2){
    // for(int i=index;i<receive_len;i+=blockDim.x)
    {
      // printf("before change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("cal env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld------%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2,BCC_DBX);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      if(sector_id < 4) {
        d_buffer[index].z += sub_box_lattice_size_2;
      }else{
        d_buffer[index].z -= sub_box_lattice_size_2+ (neighbour_local_sub_box_2 - sub_box_lattice_size_2);
      }
      z = d_buffer[index].z;

      id = getId(x,y,z);
      //printf("change pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);

      // printf("change sur pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("original sur pos:(%ld,%ld,%ld)-%ld\t	,change sur pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove_atom_op(g_V_Hash,id);
        // printf("--V--");
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove_atom_op(g_Re_Hash,id);
        //  printf("--Re--");
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove_atom_op(g_MoMo_Hash,id);
        //  printf("--MoMo--");
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove_atom_op(g_MoRe_Hash,id);
        //  printf("--MoRe--");
      }
      // printf("%ld\n",d_buffer[index].type);

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert_atom_op(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert_atom_op(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert_atom_op(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert_atom_op(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }

  }

  
}
__global__ void update_hash_set2(ChangeLattice_GPU *d_buffer,int receive_len,const int dimension,
                                int next_id,
                                long int sub_box_lattice_size_0,long int sub_box_lattice_size_1,long int sub_box_lattice_size_2,
                                long int neighbour_local_sub_box_0,long int neighbour_local_sub_box_1,long int neighbour_local_sub_box_2
){

  // printf("env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2);
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<KMC_HASH_TILE_SIZE>(block);

  long int BCC_DBX = 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // if(index >= receive_len)
  //   return;

  bool active = (index < receive_len);

  long int x=0,y=0,z=0;
  long int id=0,o_id=0;
  long int o_x=0,o_y=0,o_z=0;
  int type = 0;
  if(active){
    o_x= d_buffer[index].x;
    o_y= d_buffer[index].y;
    o_z= d_buffer[index].y;
    o_id = getId(d_buffer[index].x,d_buffer[index].y,d_buffer[index].z);
    type = d_buffer[index].type;

    if(dimension == 0){
      if(next_id % 2 == 1){
        d_buffer[index].x += BCC_DBX * sub_box_lattice_size_0;
      }else{
        d_buffer[index].x -= BCC_DBX * (sub_box_lattice_size_0 + (neighbour_local_sub_box_0 - sub_box_lattice_size_0));
      }
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      z = d_buffer[index].z;
      id = getId(x,y,z);

      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	,change gho pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
    }
    else if(dimension == 1){
      x = d_buffer[index].x;
      if(next_id == 2 || next_id == 3 || next_id == 6 || next_id == 7) {
        d_buffer[index].y += sub_box_lattice_size_1;
      } else {
        d_buffer[index].y -= sub_box_lattice_size_1 + (neighbour_local_sub_box_1 - sub_box_lattice_size_1);
      }
      y = d_buffer[index].y;
      z = d_buffer[index].z;

      id = getId(x,y,z);
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	,change gho pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
    }
    else if(dimension == 2){
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      if(next_id > 3 ) {
        d_buffer[index].z += sub_box_lattice_size_2;
      }else{
        d_buffer[index].z -= sub_box_lattice_size_2+ (neighbour_local_sub_box_2 - sub_box_lattice_size_2);
      }
      z = d_buffer[index].z;

      id = getId(x,y,z);
       //printf("change pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	,change gho pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
    }
  }
    // L1: 拦截 Buffer 中传来的重复 ID
    resolve_intra_tile_conflicts_single(id, active, tile);

    // L2: 拦截跨 Block 的并发修改
    bool lock_acquired = false;
    if (active) {
      lock_acquired = try_acquire_bitmap_lock_single(id);
      if (!lock_acquired) active = false; // 抢锁失败，本轮不操作 (可交由下一帧重试，或直接丢弃因为必定有同伴正在更新它)
    }
   
    bool need_del = active;
    if (tiled_remove_op(g_Re_Hash, id, tile, need_del,g_Re_Hash->num_buckets) == 1) {
        need_del = false; 
    }
    if (tiled_remove_op(g_MoRe_Hash, id, tile, need_del,g_MoRe_Hash->num_buckets) == 1) {
        need_del = false; 
    }
    if (tiled_remove_op(g_MoMo_Hash, id, tile, need_del, g_MoMo_Hash->num_buckets) == 1) {
        need_del = false; 
    }
    tiled_remove_op(g_V_Hash, id, tile, need_del, g_V_Hash->num_buckets);

    bool insert_V    = (active && type == LatticeTypes::V);
    bool insert_Re   = (active && type == LatticeTypes::Re);
    bool insert_MoMo = (active && type == LatticeTypes::MoMo);
    bool insert_MoRe = (active && type == LatticeTypes::MoRe);

    tiled_insert_op(g_V_Hash, id, tile, insert_V,g_V_Hash->num_buckets);
    tiled_insert_op(g_Re_Hash, id, tile, insert_Re,g_Re_Hash->num_buckets);
    tiled_insert_op(g_MoMo_Hash, id, tile, insert_MoMo,g_MoMo_Hash->num_buckets);
    tiled_insert_op(g_MoRe_Hash, id, tile, insert_MoRe, g_MoRe_Hash->num_buckets);
    if (lock_acquired) {
        release_bitmap_lock_single(id);
    }
    return;
}

__global__ void update_hash_set2_tiled(ChangeLattice_GPU *d_buffer,int receive_len,const int dimension,
                                int next_id,
                                long int sub_box_lattice_size_0,long int sub_box_lattice_size_1,long int sub_box_lattice_size_2,
                                long int neighbour_local_sub_box_0,long int neighbour_local_sub_box_1,long int neighbour_local_sub_box_2
){

  // printf("env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2);
  long int BCC_DBX = 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= receive_len)
    return;

  long int x,y,z;
  long int id=0,o_id=0;
  long int o_x=0,o_y=0,o_z=0;
  // long int id=0;
  o_x= d_buffer[index].x;
  o_y= d_buffer[index].y;
  o_z= d_buffer[index].y;
  o_id = getId(d_buffer[index].x,d_buffer[index].y,d_buffer[index].z);
  if(dimension == 0){
    // for(int i=index;i<receive_len;i+=blockDim.x)
    {
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("before change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("cal env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld------%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2,BCC_DBX);
      if(next_id % 2 == 1){
        d_buffer[index].x += BCC_DBX * sub_box_lattice_size_0;
      }else{
        d_buffer[index].x -= BCC_DBX * (sub_box_lattice_size_0 + (neighbour_local_sub_box_0 - sub_box_lattice_size_0));
      }
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      z = d_buffer[index].z;
      id = getId(x,y,z);
      // printf("after change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      //printf("change pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("change gho pos:(%ld,%ld,%ld)-%ld-%ld\n",x,y,z,id,d_buffer[index].type);
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	,change gho pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove_atom_op(g_V_Hash,id);
        // printf("--V--");
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove_atom_op(g_Re_Hash,id);
        //  printf("--Re--");
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove_atom_op(g_MoMo_Hash,id);
        // printf("--MoMo--");
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove_atom_op(g_MoRe_Hash,id);
        //  printf("--MoRe--");
      }
      // printf("%ld\n",d_buffer[index].type);

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert_atom_op(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert_atom_op(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert_atom_op(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert_atom_op(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }

  }
  else if(dimension == 1){
    // for(int i=index;i<receive_len;i+=blockDim.x)
    {
      // printf("before change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("cal env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld------%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2,BCC_DBX);
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      x = d_buffer[index].x;
      if(next_id == 2 || next_id == 3 || next_id == 6 || next_id == 7) {
        d_buffer[index].y += sub_box_lattice_size_1;
      } else {
        d_buffer[index].y -= sub_box_lattice_size_1 + (neighbour_local_sub_box_1 - sub_box_lattice_size_1);
      }
      y = d_buffer[index].y;
      z = d_buffer[index].z;

      id = getId(x,y,z);
       //printf("change pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("change gho pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	,change gho pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove_atom_op(g_V_Hash,id);
        // printf("--V--");
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove_atom_op(g_Re_Hash,id);
        //  printf("--Re--");
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove_atom_op(g_MoMo_Hash,id);
        //  printf("--MoMo--");
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove_atom_op(g_MoRe_Hash,id);
        //  printf("--MoRe--");
      }
      // printf("%ld\n",d_buffer[index].type);

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert_atom_op(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert_atom_op(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert_atom_op(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert_atom_op(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }
  }
  else if(dimension == 2){
    // for(int i=index;i<receive_len;i+=blockDim.x)
    {
      // printf("before change pos:(%ld,%ld,%ld)-%ld\n",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      // printf("cal env params:\t%ld\t%ld\t%ld ----- %ld\t%ld\t%ld------%ld\n",sub_box_lattice_size_0,sub_box_lattice_size_1,sub_box_lattice_size_2,
        // neighbour_local_sub_box_0,neighbour_local_sub_box_1,neighbour_local_sub_box_2,BCC_DBX);
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	",d_buffer[index].x,d_buffer[index].y,d_buffer[index].z,id);
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      if(next_id > 3 ) {
        d_buffer[index].z += sub_box_lattice_size_2;
      }else{
        d_buffer[index].z -= sub_box_lattice_size_2+ (neighbour_local_sub_box_2 - sub_box_lattice_size_2);
      }
      z = d_buffer[index].z;

      id = getId(x,y,z);
       //printf("change pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);
      // printf("original gho pos:(%ld,%ld,%ld)-%ld\t	,change gho pos:(%ld,%ld,%ld)-%ld\n",o_x,o_y,o_z,o_id,x,y,z,id);
      // printf("change gho pos:(%ld,%ld,%ld)-%ld\n",x,y,z,id);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove_atom_op(g_V_Hash,id);
        // printf("--V--");
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove_atom_op(g_Re_Hash,id);
        //  printf("--Re--");
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove_atom_op(g_MoMo_Hash,id);
        //  printf("--MoMo--");
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove_atom_op(g_MoRe_Hash,id);
        //  printf("--MoRe--");
      }
      // printf("%ld\n",d_buffer[index].type);

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert_atom_op(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert_atom_op(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert_atom_op(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert_atom_op(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }
  }
  
}
void transferBufferToGPU(ChangeLattice *buffer, int receive_len, const int dimension,
                          const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int id){
using namespace gpuSimulate;
    ChangeLattice_GPU *h_buffer;

      hipHostMalloc((void**)&h_buffer,receive_len*sizeof(ChangeLattice_GPU));
      init_ChangeLattice_GPU(buffer,h_buffer,receive_len);

      ChangeLattice_GPU *d_buffer;
      HANDLE_HIP(hipMalloc((void**)&d_buffer,sizeof(ChangeLattice_GPU)*receive_len));
      const double h2d_start = MPI_Wtime();
      HANDLE_HIP(hipMemcpy(d_buffer,h_buffer,sizeof(ChangeLattice_GPU)*receive_len,hipMemcpyHostToDevice));
      gpuSimulate::comm_gpu_maintenance_profiler.h2d_time += MPI_Wtime() - h2d_start;
      const double hash_update_start = MPI_Wtime();
      int blockSize = 128;
      int blockNum = (receive_len)/blockSize + 1;
      if (d_Global_Bitmap_Lock != nullptr && bitmap_lock_bytes > 0) {
        HANDLE_HIP(hipMemset(d_Global_Bitmap_Lock, 0, bitmap_lock_bytes));
      }
      //printf("start kernel update\n");
      hipLaunchKernelGGL(update_hash_set, blockNum, blockSize, 0, 0, 
        d_buffer,
        receive_len,
        dimension,
        id,
        sub_box_lattice_size[0],
        sub_box_lattice_size[1],
        sub_box_lattice_size[2],
        neighbour_local_sub_box[0],
        neighbour_local_sub_box[1],
        neighbour_local_sub_box[2]
      );
      hipDeviceSynchronize();
      gpuSimulate::comm_gpu_maintenance_profiler.hash_update_time += MPI_Wtime() - hash_update_start;
    hipFree(d_buffer);
    hipHostFree(h_buffer);
}


void transferBufferToGPU2(ChangeLattice *buffer, int receive_len, const int dimension,
                          const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int next_id){
using namespace gpuSimulate;
    ChangeLattice_GPU *h_buffer;
    hipHostMalloc((void**)&h_buffer,receive_len*sizeof(ChangeLattice_GPU));
    init_ChangeLattice_GPU(buffer,h_buffer,receive_len);

    ChangeLattice_GPU *d_buffer;
    HANDLE_HIP(hipMalloc((void**)&d_buffer,sizeof(ChangeLattice_GPU)*receive_len));
    const double h2d_start = MPI_Wtime();
    HANDLE_HIP(hipMemcpy(d_buffer,h_buffer,sizeof(ChangeLattice_GPU)*receive_len,hipMemcpyHostToDevice));
    gpuSimulate::comm_gpu_maintenance_profiler.h2d_time += MPI_Wtime() - h2d_start;
    const double hash_update_start = MPI_Wtime();

    if (d_Global_Bitmap_Lock != nullptr && bitmap_lock_bytes > 0) {
        HANDLE_HIP(hipMemset(d_Global_Bitmap_Lock, 0, bitmap_lock_bytes));
    }
    int blockSize = 128;
    int blockNum = (receive_len)/blockSize + 1;
    // hipLaunchKernelGGL(print_g_Pair_Atoms,1,1,0,0,arr_size);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(update_hash_set2,blockNum, blockSize, 0, 0, 
      d_buffer,
      receive_len,
      dimension,
      next_id,
      sub_box_lattice_size[0],
      sub_box_lattice_size[1],
      sub_box_lattice_size[2],
      neighbour_local_sub_box[0],
      neighbour_local_sub_box[1],
      neighbour_local_sub_box[2]
    );
    hipDeviceSynchronize();
    gpuSimulate::comm_gpu_maintenance_profiler.hash_update_time += MPI_Wtime() - hash_update_start;
    hipFree(d_buffer);
    hipHostFree(h_buffer);
}

void transferBufferToGPU_Device(ChangeLattice_GPU *d_buffer, int receive_len, const int dimension,
                          const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int id){
      const double hash_update_start = MPI_Wtime();
      int blockSize = 128;
      int blockNum = (receive_len)/blockSize + 1;
      hipLaunchKernelGGL(update_hash_set, blockNum, blockSize, 0, 0, 
        d_buffer,
        receive_len,
        dimension,
        id,
        sub_box_lattice_size[0],
        sub_box_lattice_size[1],
        sub_box_lattice_size[2],
        neighbour_local_sub_box[0],
        neighbour_local_sub_box[1],
        neighbour_local_sub_box[2]
      );
      hipDeviceSynchronize();
      gpuSimulate::comm_gpu_maintenance_profiler.hash_update_time += MPI_Wtime() - hash_update_start;
}

void transferBufferToGPU2_Device(ChangeLattice_GPU *d_buffer, int receive_len, const int dimension,
                          const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int next_id){
    const double hash_update_start = MPI_Wtime();
    int blockSize = 128;
    int blockNum = (receive_len)/blockSize + 1;
    hipDeviceSynchronize();
    hipLaunchKernelGGL(update_hash_set2,blockNum, blockSize, 0, 0, 
      d_buffer,
      receive_len,
      dimension,
      next_id,
      sub_box_lattice_size[0],
      sub_box_lattice_size[1],
      sub_box_lattice_size[2],
      neighbour_local_sub_box[0],
      neighbour_local_sub_box[1],
      neighbour_local_sub_box[2]
    );
    hipDeviceSynchronize();
    gpuSimulate::comm_gpu_maintenance_profiler.hash_update_time += MPI_Wtime() - hash_update_start;
}

void rebuildPairAtomsAfterComm() {
  using namespace gpuSimulate;
  if (pair_atoms_size <= 0) return;
  if (!d_Pair_Atoms) return;
  if (!MoMo_Hash && !MoRe_Hash) return;

  const double rebuild_cleanup_start = MPI_Wtime();

  static int *d_pair_compact_count = nullptr;
  if (!d_pair_compact_count) {
    HANDLE_HIP(hipMalloc(reinterpret_cast<void**>(&d_pair_compact_count), sizeof(int)));
  }

  const int max_out = static_cast<int>(pair_atoms_size);
  HANDLE_HIP(hipMemset(d_Pair_Atoms, 0xFF, static_cast<size_t>(pair_atoms_size) * sizeof(long int)));
  HANDLE_HIP(hipMemset(d_pair_compact_count, 0, sizeof(int)));

  const int blockSize = 256;
  if (MoMo_Hash && MoMo_Hash->device_ptr()) {
    const int cap = MoMo_Hash->get_capacity();
    const int grid = (cap + blockSize - 1) / blockSize;
    if (grid > 0) {
      hipLaunchKernelGGL(compactHashSetBoundedKernel, grid, blockSize, 0, 0,
                         MoMo_Hash->device_ptr(), d_Pair_Atoms, d_pair_compact_count, max_out);
    }
  }
  if (MoRe_Hash && MoRe_Hash->device_ptr()) {
    const int cap = MoRe_Hash->get_capacity();
    const int grid = (cap + blockSize - 1) / blockSize;
    if (grid > 0) {
      hipLaunchKernelGGL(compactHashSetBoundedKernel, grid, blockSize, 0, 0,
                         MoRe_Hash->device_ptr(), d_Pair_Atoms, d_pair_compact_count, max_out);
    }
  }

  hipDeviceSynchronize();
  gpuSimulate::comm_gpu_maintenance_profiler.rebuild_cleanup_time += MPI_Wtime() - rebuild_cleanup_start;
}

__global__ void packChangeLatticeFromIdsKernel(const _type_lattice_id *ids, ChangeLattice_GPU *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  _type_lattice_id id = ids[i];
  _type_lattice_id x = id % g_meta->size_x;
  _type_lattice_id y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_id z = id / (g_meta->size_x * g_meta->size_y);
  out[i].x = x;
  out[i].y = y;
  out[i].z = z;
  out[i].type = static_cast<int>(getType(id));
}

void packChangeLatticeFromIds_Device(const _type_lattice_id *d_ids, ChangeLattice_GPU *d_out, int n) {
  int blockSize = 256;
  int blockNum = (n + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(packChangeLatticeFromIdsKernel, blockNum, blockSize, 0, 0, d_ids, d_out, n);
  hipDeviceSynchronize();
}
