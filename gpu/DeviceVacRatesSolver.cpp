#include "../src/lattice/lattice_types.h"
#include "../src/type_define.h"
#include "DeviceVacRatesSolver.h"
#include <unordered_map>
#include <unordered_set>
#include <hip/hip_runtime.h>
#include <iostream>


// **构造函数：从 unordered_set 初始化**
HIPHashSet::HIPHashSet(const std::unordered_set<long int>& cpu_set) {
    int set_size = cpu_set.size();
    if (set_size == 0) set_size = 1;

    h_table.capacity = set_size * 2;  // 保持装载因子 ~0.5
    h_table.empty_flag = INIT_FLAG;
    h_table.tombstone_flag = TOMBSTONE;

    // **分配主机端数组，并初始化**
    long int* h_keys = new long int[h_table.capacity];
    memset(h_keys, INIT_FLAG, h_table.capacity * sizeof(long int));

    // **填充哈希表**
    for (const auto& key : cpu_set) {
        int index = key % h_table.capacity;
        while (h_keys[index] != INIT_FLAG && h_keys[index] != TOMBSTONE) {
            index = (index + 1) % h_table.capacity;
        }
        h_keys[index] = key;
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
    int set_size = cpu_set.size() + size;
    if (set_size == 0) set_size = 1;

    h_table.capacity = set_size * 2;  // 保持装载因子 ~0.5
    h_table.empty_flag = INIT_FLAG;
    h_table.tombstone_flag = TOMBSTONE;

    // **分配主机端数组，并初始化**
    long int* h_keys = new long int[h_table.capacity];
    memset(h_keys, INIT_FLAG, h_table.capacity * sizeof(long int));

    // **填充哈希表**
    for (const auto& key : cpu_set) {
        int index = key % h_table.capacity;
        while (h_keys[index] != INIT_FLAG && h_keys[index] != TOMBSTONE) {
            index = (index + 1) % h_table.capacity;
        }
        h_keys[index] = key;
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
  h_table.capacity = size * 2;
  h_table.empty_flag = INIT_FLAG;
  h_table.tombstone_flag = TOMBSTONE;

  // 分配主机端 keys 数组，并初始化为 INIT_FLAG
  long int* h_keys = new long int[h_table.capacity];
  memset(h_keys, INIT_FLAG, h_table.capacity * sizeof(long int));

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

    printf("------this is buffer data-------------\n");
    for(int i=0;i<len;i++){
        h_buffer[i].x = buffer[i].x;
        h_buffer[i].y = buffer[i].y;
        h_buffer[i].z = buffer[i].z;
        h_buffer[i].type = buffer[i].type._type;
        printf("h_buffer data (%ld,%ld,%ld)-%ld\t",h_buffer[i].x,h_buffer[i].y,h_buffer[i].z,h_buffer[i].type);
        // if(i%10 == 0)
        // printf("\n");
    }
     // printf("\n");
    printf("------ end buffer data----------------\n");

    return 1;
}

// 清空HIPHashSet中的所有元素，保留哈希表结构
void HIPHashSet::clear() {
    // 确保设备端结构体已分配
    if (d_table == nullptr) return;

    // 1. 先将主机端h_table中的keys指针设为nullptr（避免误操作）
    long int* temp_keys = h_table.keys;
    h_table.keys = nullptr;

    // 2. 分配临时主机端数组用于初始化
    long int* h_init_keys = new long int[h_table.capacity];
    std::fill(h_init_keys, h_init_keys + h_table.capacity, INIT_FLAG);

    // 3. 将初始化数据拷贝到设备端keys数组
    hipError_t err = hipMemcpy(temp_keys, h_init_keys, 
                               h_table.capacity * sizeof(long int), 
                               hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        delete[] h_init_keys;
        return;
    }

    // 4. 释放临时主机端数组
    delete[] h_init_keys;

    // 5. 更新主机端结构体（可选，因为实际操作在设备端）
    h_table.keys = temp_keys;
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
        if (key != INIT_FLAG && key != TOMBSTONE) {
            cpu_hash.emplace(key);  // 值设为true表示存在
           hash_num++;
        }
    }
     printf("this key num is %ld\n",hash_num);
     hipDeviceSynchronize();
    // HANDLE_HIP(hipMemcpy(h_ghost_Hash, ghost_Hash->device_ptr()->keys, ghost_Hash->device_ptr()->capacity*sizeof(long int), hipMemcpyDeviceToHost));
}
