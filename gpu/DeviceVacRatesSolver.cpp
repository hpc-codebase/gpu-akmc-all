#include "../src/lattice/lattice_types.h"
#include "../src/type_define.h"
#include "DeviceVacRatesSolver.h"
#include <unordered_map>
#include <unordered_set>
#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include <limits>

namespace {
int compute_hash_capacity(int requested_slots) {
    if (requested_slots <= 0) requested_slots = 1;
    // 开放寻址必须保留空槽，探测链才能在“第一个 empty”处自然终止，
    // 故按固定 2 倍（负载因子 0.5）预留容量，直接写死、不做可调参数。
    long long capacity = static_cast<long long>(requested_slots) * 2;
    if (capacity < BUCKET_SIZE) capacity = BUCKET_SIZE;
    long long remainder = capacity % BUCKET_SIZE;
    if (remainder != 0) {
        capacity += (BUCKET_SIZE - remainder);
    }
    if (capacity > std::numeric_limits<int>::max()) {
        std::cerr << "Fatal Error: GPU hash capacity exceeds int range: "
                  << capacity << std::endl;
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(capacity);
}
}


// **构造函数：从 unordered_set 初始化**
HIPHashSet::HIPHashSet(const std::unordered_set<long int>& cpu_set) {
    int set_size = cpu_set.size();
    if (set_size == 0) set_size = 1;

    h_table.capacity = compute_hash_capacity(set_size);
    h_table.empty_flag = GPU_HASH_EMPTY_KEY;
    h_table.tombstone_flag = GPU_HASH_TOMBSTONE_KEY;
    h_table.num_buckets = h_table.capacity / BUCKET_SIZE;
    // 初始化哈希函数,使用固定种子 2
    h_table.hasher = DeviceHasher(2);

    // **分配主机端数组，并初始化**
    long int* h_keys = new long int[h_table.capacity];
    memset(h_keys, GPU_HASH_EMPTY_KEY, h_table.capacity * sizeof(long int));

    // **填充哈希表**
    for (const auto& key : cpu_set) {
        unsigned int loop_count = 0;
        const unsigned int max_probe = hash_effective_max_probes(h_table.num_buckets);
        unsigned int buck_num = hash_probe_bucket(h_table.num_buckets, key, loop_count);
        bool inserted = false;

        while (loop_count <= max_probe) {
            int base_index = buck_num * BUCKET_SIZE;    
            
            // 在当前桶内的 64 个槽位中找空位
            for(int i = 0; i < BUCKET_SIZE; i++) {
                if(h_keys[base_index + i] == GPU_HASH_EMPTY_KEY || h_keys[base_index + i] == GPU_HASH_TOMBSTONE_KEY) {
                    h_keys[base_index + i] = key;
                    inserted = true;
                    break;
                }
            }
            if (inserted) break; // 插入成功，处理下一个 key

            // 桶满了，线性探测下一个桶
            loop_count++;
            buck_num = hash_probe_bucket(h_table.num_buckets, key, loop_count);
        }
        if (!inserted) {
            std::cerr << "Fatal Error: CPU Hash Table insertion exceeded probe limit "
                      << max_probe << " for key " << key << std::endl;
        }
    }

    // **分配设备端 keys 数组**
    hipError_t err = hipMalloc(&h_table.keys, h_table.capacity * sizeof(long int));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
        delete[] h_keys;
        return;
    }

    // **拷贝数据到设备**
    err = hipMemcpy(h_table.keys, h_keys, h_table.capacity * sizeof(long int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        delete[] h_keys;
        return;
    }

    // **分配 GPU 上的结构体**
    err = hipMalloc(&d_table, sizeof(GPUHashSet));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
        return;
    }

    // **拷贝结构体到 GPU**
    err = hipMemcpy(d_table, &h_table, sizeof(GPUHashSet), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy (GPUHashSet) failed: " << hipGetErrorString(err) << std::endl;
        return;
    }

    delete[] h_keys;
}
HIPHashSet::HIPHashSet(const std::unordered_set<long int>& cpu_set,int size){
    // size 语义：目标能容纳的 key 总数（含初始元素与增长余量），由调用方显式给定。
    // 注意不再累加 cpu_set.size()：此前把调用方已算好的总量又加一次，导致
    // MoMo 表（cpu_set 非空）比 MoRe 表（cpu_set 为空）多开了一倍容量。
    int set_size = size;
    if (set_size < (int)cpu_set.size()) set_size = (int)cpu_set.size();
    if (set_size == 0) set_size = 1;

    h_table.capacity = compute_hash_capacity(set_size);
    h_table.empty_flag = GPU_HASH_EMPTY_KEY;
    h_table.tombstone_flag = GPU_HASH_TOMBSTONE_KEY;
    h_table.num_buckets = h_table.capacity / BUCKET_SIZE;

    // **分配主机端数组，并初始化**
    long int* h_keys = new long int[h_table.capacity];
    memset(h_keys, GPU_HASH_EMPTY_KEY, h_table.capacity * sizeof(long int));

    // 初始化哈希函数，使用固定种子 2
    h_table.hasher = DeviceHasher(2);

    for (const auto& key : cpu_set) {
        unsigned int loop_count = 0;
        const unsigned int max_probe = hash_effective_max_probes(h_table.num_buckets);
        unsigned int buck_num = hash_probe_bucket(h_table.num_buckets, key, loop_count);
        bool inserted = false;

        while (loop_count <= max_probe) {
            int base_index = buck_num * BUCKET_SIZE;    
            
            // 在当前桶内的 64 个槽位中找空位
            for(int i = 0; i < BUCKET_SIZE; i++) {
                if(h_keys[base_index + i] == GPU_HASH_EMPTY_KEY || h_keys[base_index + i] == GPU_HASH_TOMBSTONE_KEY) {
                    h_keys[base_index + i] = key;
                    inserted = true;
                    break;
                }
            }
            if (inserted) break; // 插入成功，处理下一个 key

            // 桶满了，线性探测下一个桶
            loop_count++;
            buck_num = hash_probe_bucket(h_table.num_buckets, key, loop_count);
        }
        if (!inserted) {
            std::cerr << "Fatal Error: CPU Hash Table insertion exceeded probe limit "
                      << max_probe << " for key " << key << std::endl;
        }
    }

    // **分配设备端 keys 数组**
    hipError_t err = hipMalloc(&h_table.keys, h_table.capacity * sizeof(long int));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
        delete[] h_keys;
        return;
    }

    // **拷贝数据到设备**
    err = hipMemcpy(h_table.keys, h_keys, h_table.capacity * sizeof(long int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        delete[] h_keys;
        return;
    }

    // **分配 GPU 上的结构体**
    err = hipMalloc(&d_table, sizeof(GPUHashSet));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
        return;
    }

    // **拷贝结构体到 GPU**
    err = hipMemcpy(d_table, &h_table, sizeof(GPUHashSet), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy (GPUHashSet) failed: " << hipGetErrorString(err) << std::endl;
        return;
    }

    delete[] h_keys;

}
// **构造函数：指定容量**
HIPHashSet::HIPHashSet(int size) {
  // 设置容量为 size * 2，保持装载因子约 0.5
  h_table.capacity = compute_hash_capacity(size);
  h_table.empty_flag = GPU_HASH_EMPTY_KEY;
  h_table.tombstone_flag = GPU_HASH_TOMBSTONE_KEY;
  h_table.num_buckets = h_table.capacity / BUCKET_SIZE;

  h_table.hasher = DeviceHasher(2); // 使用固定种子 2

  // 分配主机端 keys 数组，并初始化为 INIT_FLAG
  long int* h_keys = new long int[h_table.capacity];
  memset(h_keys, GPU_HASH_EMPTY_KEY, h_table.capacity * sizeof(long int));

  // 分配设备端 keys 数组
  hipError_t err = hipMalloc(&h_table.keys, h_table.capacity * sizeof(long int));
  if (err != hipSuccess) {
      std::cerr << "hipMalloc failed for keys: " << hipGetErrorString(err) << std::endl;
      delete[] h_keys;
      return;
  }

  // 拷贝主机端的 keys 数组到设备
  err = hipMemcpy(h_table.keys, h_keys, h_table.capacity * sizeof(long int), hipMemcpyHostToDevice);
  if (err != hipSuccess) {
      std::cerr << "hipMemcpy failed for keys: " << hipGetErrorString(err) << std::endl;
      delete[] h_keys;
      return;
  }

  // 分配 GPU 上的结构体（设备端结构体指针）
  err = hipMalloc(&d_table, sizeof(GPUHashSet));
  if (err != hipSuccess) {
      std::cerr << "hipMalloc failed for GPUHashSet: " << hipGetErrorString(err) << std::endl;
      delete[] h_keys;
      return;
  }

  // 拷贝主机端的 h_table 结构体到设备
  err = hipMemcpy(d_table, &h_table, sizeof(GPUHashSet), hipMemcpyHostToDevice);
  if (err != hipSuccess) {
      std::cerr << "hipMemcpy failed for GPUHashSet: " << hipGetErrorString(err) << std::endl;
      // 注意：此时 d_table 内存已经分配，需要后续释放（析构函数中会处理）
      delete[] h_keys;
      return;
  }

  delete[] h_keys;
}

// **析构函数**
HIPHashSet::~HIPHashSet() {
    if (h_table.keys) {
        hipFree(h_table.keys);
    }
    if (d_table) {
        hipFree(d_table);
    }
}


int init_ChangeLattice_GPU(ChangeLattice *buffer,ChangeLattice_GPU *h_buffer,int len){

    // printf("------this is buffer data-------------\n");
    for(int i=0;i<len;i++){
        h_buffer[i].x = buffer[i].x;
        h_buffer[i].y = buffer[i].y;
        h_buffer[i].z = buffer[i].z;
        h_buffer[i].type = buffer[i].type._type;
        // printf("h_buffer data (%ld,%ld,%ld)-%ld\t",h_buffer[i].x,h_buffer[i].y,h_buffer[i].z,h_buffer[i].type);
        // if(i%10 == 0)
        // printf("\n");
    }
     // printf("\n");
    // printf("------ end buffer data----------------\n");

    return 1;
}

// 清空HIPHashSet中的所有元素，保留哈希表结构
void HIPHashSet::clear() {
    if (d_table == nullptr) return;
    if (h_table.keys == nullptr) return;
    hipError_t err = hipMemset(h_table.keys, 0xFF, h_table.capacity * sizeof(long int));
    if (err != hipSuccess) {
        std::cerr << "hipMemset failed: " << hipGetErrorString(err) << std::endl;
        return;
    }
}

void HIPHashSet::copyToHost(std::unordered_set<_type_lattice_id> &cpu_hash){

     // 创建临时向量存储GPU数据
    std::vector<_type_lattice_id> keys(h_table.capacity);
    
    // 检查GPU指针有效性
    if (!h_table.keys) {
        std::cerr << "Error: GPU key array pointer is null!" << std::endl;
        return;
    }
    
    // 将GPU数据复制到CPU向量
    hipError_t err = hipMemcpy(
        keys.data(), 
        h_table.keys, 
        h_table.capacity * sizeof(_type_lattice_id),
        hipMemcpyDeviceToHost
    );
    
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        return;
    }
    
    // 同步确保传输完成
    hipDeviceSynchronize();
    
    // 过滤有效键并插入到CPU哈希表
    long int hash_num =0;
    cpu_hash.clear();
    for (const auto& key : keys) {
        if (key != GPU_HASH_EMPTY_KEY && key != GPU_HASH_TOMBSTONE_KEY) {
            cpu_hash.emplace(key);  // 值设为true表示存在
           hash_num++;
        }
    }
    //  printf("this key num is %ld\n",hash_num);
     hipDeviceSynchronize();
    // HANDLE_HIP(hipMemcpy(h_ghost_Hash, ghost_Hash->device_ptr()->keys, ghost_Hash->device_ptr()->capacity*sizeof(long int), hipMemcpyDeviceToHost));
}
