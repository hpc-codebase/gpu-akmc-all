#ifndef DEVICEVACRATESSOVLER_H
#define DEVICEVACRATESSOVLER_H

#define __HIP_PLATFORM_AMD__   

#include "../src/lattice/lattice_types.h"
#include "../src/lattice/lattice.h"
// #include "gpu_simulate.h"
#include <hip/hip_runtime.h>
#include "gpuError.h"
#include "../src/type_define.h"
#include <unordered_set>

#define NN_TOTAL 126
#define RE_SCALE_SIZE 1.2
#define STREAM_SIZE 4
#ifndef KMC_HASH_BUCKET_SIZE
// 桶大小可配置（编译时 -DKMC_HASH_BUCKET_SIZE=N 覆盖）：
//   1  = 最原始单线程开放寻址（每桶 1 槽，纯线性探测），作对比基准；
//   64 = 桶式开放寻址（每桶 64 槽）。做 tile vs 单线程、桶大小对比时在此切换。
#define KMC_HASH_BUCKET_SIZE 1
#endif
#define BUCKET_SIZE KMC_HASH_BUCKET_SIZE

#ifndef KMC_HASH_TILE_SIZE
#define KMC_HASH_TILE_SIZE BUCKET_SIZE
#endif

#define HASH_SEED_PRIMARY 2u

// MoRe 在一次 recb launch 内会经历大量 remove/insert churn。每次 recb 后会
// 清理墓碑，因此默认容量由 18 倍压到 10 倍，兼顾探测余量和显存/扫描开销。
#ifndef KMC_MORE_HASH_CAPACITY_MULTIPLIER
#define KMC_MORE_HASH_CAPACITY_MULTIPLIER 10
#endif
#if KMC_MORE_HASH_CAPACITY_MULTIPLIER <= 0
#error "KMC_MORE_HASH_CAPACITY_MULTIPLIER must be positive"
#endif

static constexpr long int GPU_HASH_EMPTY_KEY = -1L;
static constexpr long int GPU_HASH_TOMBSTONE_KEY = -2L;

__host__ __device__ __forceinline__ unsigned int lattice_hash_mix(long int k, unsigned int seed) {
    unsigned long long int h = (unsigned long long int)k;
    h ^= seed;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (unsigned int)h;
}

__host__ __device__ __forceinline__ unsigned int hash_h1(long int key) {
    return lattice_hash_mix(key, HASH_SEED_PRIMARY);
}

__host__ __device__ __forceinline__ unsigned int hash_primary_bucket(unsigned int num_buckets, long int key) {
    return hash_h1(key) % num_buckets;
}

__host__ __device__ __forceinline__ unsigned int hash_probe_bucket(unsigned int num_buckets, long int key, unsigned int probe) {
    return (hash_primary_bucket(num_buckets, key) + probe) % num_buckets;
}

// 标准线性探测不设人为次数上限：探测序列 (h1 + probe) % num_buckets 会遍历全部桶，
// 只要表里至少有一个空槽，查找/插入/删除就必然在 num_buckets 步内终止。
// 这里把探测上界直接设为表长（num_buckets），等价于“无限探测”的教科书语义，
// 避免人为截断造成“存在却查不到”的假阴性（这正是此前 roster/hash 失配、
// tx 失败、stale walker 的根源）。
__host__ __device__ __forceinline__ unsigned int hash_effective_max_probes(unsigned int num_buckets) {
    return num_buckets;
}

// 定义一个可以在 Device 端使用的哈希函数对象 (Functor)
struct DeviceHasher {
    unsigned int m_seed;

    __host__ __device__ DeviceHasher() : m_seed(0) {}
    __host__ __device__ DeviceHasher(unsigned int seed) : m_seed(seed) {}

    __host__ __device__ __forceinline__ unsigned int operator()(long int k) const {
        unsigned long long int h = (unsigned long long int)k;
        h ^= m_seed;
        
        // 64 位的雪崩混合运算
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        
        // 最终转回 32 位用于数组取模
        return (unsigned int)h;
    }
};
__device__ __forceinline__ long int atomic_load_relaxed(const long int* addr) {
    return *reinterpret_cast<volatile const long int*>(addr);
}

struct dev_Vacancy {
    // LatticeTypes type;
    int id;
    double rates[8];
};

struct dev_nnLattice {
    LatticeTypes type;
    int id;
};

struct dev_event {
    long int from_id;
    long int to_id;
    LatticeTypes to_type;
};

struct dev_Lattice{
    LatticeTypes::lat_type type;
    long int id;
};

struct dev_atom{
    int atom_id;
    LatticeTypes::lat_type atom_type;
    int atom12nn[14];
    LatticeTypes::lat_type atom12nn_type[14];

    int atom1nn[8];
    LatticeTypes::lat_type atom1nn_type[8];
    int atom2nn[6];
    LatticeTypes::lat_type atom2nn_type[6];
    int random_num1;
    double ranmov;

    int out_sector;
    int exchange;
    int exchange_id;
    LatticeTypes::lat_type exchange_type;

    int numRe;
    int numSIA;
};


struct dev_meta{
    _type_lattice_coord size_x, size_y, size_z;

  /**
   * \brief the lattice size of simulation box in each dimension on current
   * process. \note the box_x is two times then real box size (without ghost
   * area) due to BCC structure. box_y and box_z is the same as the real
   * lattice size (without ghost area).
   */
   _type_lattice_coord box_x, box_y, box_z;

  /**
   * \brief the lattice size of global simulation box in each dimension.
   */
  _type_lattice_coord g_box_x, g_box_y, g_box_z;

  /**
   * \brief the baseline coordinate of current process at each dimension for
   * setting global lattice id.
   */
  _type_lattice_coord g_base_x, g_base_y, g_base_z;

  /**
   * \brief the ghost lattice size of lattice lists array in each dimension
   * \note the ghost_x is two times then real lattice size.
   */
  _type_lattice_coord ghost_x, ghost_y, ghost_z;

  // the max local lattice id in box.
  int _max_id;

  // global id = local id + local_base_id
  int local_base_id = 0;

  _type_lattice_coord regions[48];
};

struct GPUHashSet {
    long int* keys;  // 存储集合元素的数组
    int capacity;  // 哈希表总容量
    long int empty_flag;  // 空槽标记值
    long int tombstone_flag;  // 删除标记值
    unsigned int num_buckets;//桶的数量
    DeviceHasher hasher;    // 哈希函数对象
};

class HIPHashSet {
private:
    GPUHashSet h_table;  // 主机端结构体
    GPUHashSet* d_table; // 设备端结构体指针
    const long int INIT_FLAG = -1;  // 空槽标记
    const long int TOMBSTONE = -2;  // 删除标记

public:
    HIPHashSet(const std::unordered_set<long int>& cpu_set);
    HIPHashSet(const std::unordered_set<long int>& cpu_set,int size);
    HIPHashSet(int size);
    void clear();
    void copyToHost(std::unordered_set<_type_lattice_id> &cpu_hash);
    ~HIPHashSet();

    // 返回设备端指针
    GPUHashSet* device_ptr() { return d_table; }
    unsigned int get_num_buckets() const { return h_table.num_buckets; }
    int get_capacity() const { return h_table.capacity; }
    
};

int init_ChangeLattice_GPU(ChangeLattice *buffer,ChangeLattice_GPU *h_buffer,int len);
#endif /* DEVICEVACRATESSOVLER_H */
