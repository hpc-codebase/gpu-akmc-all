
#include "gpu_simulate.h"

// #include <hiprand_kernel.h>
// Global Variables
__device__ static GPUHashSet *g_MoRe_Hash;
__device__ static GPUHashSet *g_MoMo_Hash = nullptr;
__device__ static GPUHashSet *g_Re_Hash;
__device__ static GPUHashSet *g_V_Hash;
__device__ static dev_meta *g_meta;
__device__ static GPUHashSet *g_Busy_Set;
__device__ static GPUHashSet *g_surface_Hash;
__device__ static GPUHashSet *g_ghost_Hash;
__device__ static long int  *g_Pair_Atoms;

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

    long int *Pair_Atoms;
    long int *d_Pair_Atoms;
    dev_meta *d_meta;



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
__device__ int hash_set_contains_debug(GPUHashSet* table, int key) {
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

// 设备端存在性检查函数
__device__ int hash_set_contains(GPUHashSet* table, long int key) {
  int i=0;
  int index = key % table->capacity;
  const int original_index = index;
  volatile long int* keys = (volatile long int*)table->keys;  // 避免编译器优化

  while (true) {
      long int current = keys[index];  // 使用 volatile 方式读取 key

      if (current == key) {
          return 1;  // 找到 key
      }
      if (current == table->empty_flag) {
          return 0;  // 空槽，说明 key 不存在
      }

      index = (index + 1) % table->capacity;
      if (index == original_index) {
          return 0;  // 完整遍历哈希表后仍未找到 key
      }
  }
}


__device__ int hash_set_find(long int key){

}
// 设备端插入函数（线程安全）
__device__ int hash_set_insert(GPUHashSet* set, long int key) {
  int pos = abs(key) % set->capacity;  // 保持修正后的哈希计算
  int start_pos = pos;
  
  do {
      long int current = set->keys[pos];  // 直接内存访问
      
      if (current == key) {
          return 0; // 已存在相同key
      }
      
      if (current == set->empty_flag || current == set->tombstone_flag) {
          set->keys[pos] = key; // 直接赋值
          return 1;
      }
      
      pos = (pos + 1) % set->capacity;
  } while (pos != start_pos);
  
  return 0; // 哈希表已满
}

// 设备端删除函数（线程安全）
__device__ int hash_set_remove(GPUHashSet* table,long int key) {
  int index = abs(key) % table->capacity;
  const int original_index = index;
  
  do {
      long int current = table->keys[index]; // 直接内存访问
      
      if (current == table->empty_flag) {
          return 0;
      }
      
      if (current == key) {
          table->keys[index] = table->tombstone_flag; // 直接赋值
          return 1;
      }
      
      index = (index + 1) % table->capacity;
  } while (index != original_index);
  
  return 0;
}

__device__ long atomicExch(long* address, long val) {
    unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long val_as_ull = static_cast<unsigned long long>(val);
    
    unsigned long long old_val = atomicExch(addr_as_ull, val_as_ull);
    return static_cast<long>(old_val);
}

// 辅助函数
__device__ unsigned long long int_as_longlong(unsigned int low, unsigned int high) {
    return ((unsigned long long)high << 32) | low;
}

// 替代CUDA的__longlong_as_int函数
// 原子加载操作 - 支持long类型
__device__ long atomicLoad(long* address) {
    unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(address);
    
    // 使用两个32位原子操作模拟64位原子加载
    unsigned int* addr_as_int = reinterpret_cast<unsigned int*>(addr_as_ull);
    unsigned int low = atomicExch(&addr_as_int[0], addr_as_int[0]);
    unsigned int high = atomicExch(&addr_as_int[1], addr_as_int[1]);
    
    return static_cast<long>(int_as_longlong(low, high));
}

// 原子交换操作 - 支持long类型

__device__ int hash_set_remove_atom_op(GPUHashSet* table, long key) {
    unsigned long long index = key % table->capacity;
    unsigned long long original_index = index;
    
    do {
        // 使用适配long类型的原子加载
        long current = atomicLoad(&table->keys[index]);
        
        if (current == table->empty_flag) {
            return 0;
        }
        
        if (current == key) {
            // 使用适配long类型的原子交换
            atomicExch(&table->keys[index], table->tombstone_flag);
            return 1;
        }
        
        index = (index + 1) % table->capacity;
    } while (index != original_index);
    
    return 0;
}

__device__ int hash_set_insert_atom_op(GPUHashSet* set, long int key) {
  unsigned long long pos = key % set->capacity;  // 保持修正后的哈希计算
  unsigned long long start_pos = pos;
  
  do {
      // long int current = set->keys[pos];  // 直接内存访问
      long current = atomicLoad(&set->keys[pos]);
      
      if (current == key) {
          return 0; // 已存在相同key
      }
      
      if (current == set->empty_flag || current == set->tombstone_flag) {
          // set->keys[pos] = key; // 直接赋值
          atomicExch(&set->keys[pos], key);
          return 1;
      }
      
      pos = (pos + 1) % set->capacity;
  } while (pos != start_pos);
  
  return 0; // 哈希表已满
}

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

void selectAndPerformEventGPU(double excepted_rand, int rank, _type_lattice_count step, int sect,
                              std::array<std::unordered_set<_type_lattice_id>, 7>& exchange_ghost, const unsigned int sector_id,
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_x, 
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_y,
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_z,
                              std::array<comm::Region<comm::_type_lattice_size>, 56>& ghost_region){
  using namespace gpuSimulate;

  blockSize = 1;
  numBlocks = 1;
  hipLaunchKernelGGL(selectEventGPU, numBlocks, blockSize, 0, 0, 
                    deviceVacancyInfo,
                    deviceNeighbourInfo,
                    d_event,
                    currentVacCount,
                     *sum_rates * excepted_rand);

  CHECK_FOR_ERROR();

  HANDLE_HIP(hipMemcpy(h_event, d_event, sizeof(dev_event), hipMemcpyDeviceToHost));
  // kiwi::logs::v(" ", " excepted_rate is : {} step is : {} sect is : {} from_id is : {} to_id is : {}. \n", *sum_rates * excepted_rand, step, sect, h_event->from_id, h_event->to_id);
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
                            long int *Pair_Atoms
                            ){
  if (threadIdx.x == 0 && blockIdx.x == 0) {
  g_meta = meta;
  g_MoRe_Hash = MoRe_Hash;
  g_MoMo_Hash = MoMo_Hash;
  printf("MoMo:%p\n",MoMo_Hash);
  printf("g_MoMo:%p\n",g_MoMo_Hash);
  g_Re_Hash = Re_Hash;
  g_V_Hash = V_Hash;
  g_Busy_Set = Busy_Set_Hash;
  g_surface_Hash = surface_Hash;
  g_ghost_Hash = ghost_Hash;
  g_Pair_Atoms = Pair_Atoms;
  
    // printHashTableKernel(g_MoMo_Hash,200,false);
    printDevMetaKernel(g_meta);

   
  }
}
__global__ void printpointer(){
  if(g_MoMo_Hash == nullptr)
    printf("is a null\n");
  else
    printf("not null\n");
}
__device__ LatticeTypes::lat_type getType(const long int& id){

      if(hash_set_contains(g_MoRe_Hash,id))       return LatticeTypes::MoRe;
      else if(hash_set_contains(g_MoMo_Hash,id))  return LatticeTypes::MoMo;
      else if(hash_set_contains(g_Re_Hash,id))    return LatticeTypes::Re;
      else if(hash_set_contains(g_V_Hash,id))     return LatticeTypes::V;
      else return LatticeTypes::Mo;

}

__device__ _type_lattice_id getId(_type_lattice_size x,_type_lattice_size y,_type_lattice_size z){
  return x + g_meta->size_x * (y + z * g_meta->size_y);
}
__device__ void getnnlattice(_type_lattice_id latti_id, dev_Lattice *_1nn_list_lattice){
  _1nn_list_lattice->id = latti_id;
  if(hash_set_contains(g_MoRe_Hash,latti_id))        _1nn_list_lattice->type = LatticeTypes::MoRe;
  else if(hash_set_contains(g_MoMo_Hash,latti_id))   _1nn_list_lattice->type = LatticeTypes::MoMo;
  else if(hash_set_contains(g_Re_Hash,latti_id))     _1nn_list_lattice->type = LatticeTypes::Re;
  else if(hash_set_contains(g_V_Hash,latti_id))      _1nn_list_lattice->type = LatticeTypes::V;
  else _1nn_list_lattice->type = LatticeTypes::Mo;

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

__device__ int isSurfaceLat(_type_lattice_id lid){
   _type_lattice_coord x = 0, y = 0, z = 0;
    getCoordByLId(lid, &x, &y, &z);
    x -= g_meta->ghost_x;
    y -= g_meta->ghost_y;
    z -= g_meta->ghost_z;
    return (0 <= x && x < g_meta->ghost_x) ||(0 <= y && y < g_meta->ghost_y) || (0 <= z && z < g_meta->ghost_z) || (x >= (g_meta->box_x - g_meta->ghost_x) && x < g_meta->box_x) || (y >= (g_meta->box_y - g_meta->ghost_y) && y < g_meta->box_y) || (z >= (g_meta->box_z - g_meta->ghost_z) && z < g_meta->box_z);
}

__global__ void calcExchangePairs_test(long int *Pair_Atoms,int count,
    dev_meta *meta,
    GPUHashSet *MoRe_Hash,GPUHashSet *MoMo_Hash,
    GPUHashSet *Re_Hash,GPUHashSet *V_Hash,
    GPUHashSet *ghost_Hash,GPUHashSet *surface_Hash,
    GPUHashSet *g_Busy_Set){
    _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= count)
      return;    
    _type_lattice_id id = Pair_Atoms[tid];
      // 1. 插入测试
      int test_key = tid; // 用线程ID作为测试的键
      int insert_result = hash_set_insert_atom_op(g_Busy_Set, test_key);
      if (insert_result == 1) {
          printf("Thread %d: Successfully inserted key %d into g_Busy_Set.\n", tid, test_key);
      } else {
          printf("Thread %d: Failed to insert key %d into g_Busy_Set.\n", tid, test_key);
      }

      // 2. 查找测试
      int contains_result = hash_set_contains(g_Busy_Set, test_key);
      if (contains_result == 1) {
          printf("Thread %d: Key %d found in g_Busy_Set.\n", tid, test_key);
      } else {
          printf("Thread %d: Key %d NOT found in g_Busy_Set.\n", tid, test_key);
      }

      // 3. 删除测试
      int remove_result = hash_set_remove_atom_op(g_Busy_Set, test_key);
      if (remove_result == 1) {
          printf("Thread %d: Successfully removed key %d from g_Busy_Set.\n", tid, test_key);
      } else {
          printf("Thread %d: Failed to remove key %d from g_Busy_Set.\n", tid, test_key);
      }

      // 验证删除后是否仍然存在
      contains_result = hash_set_contains(g_Busy_Set, test_key);
      if (contains_result == 1) {
          printf("Thread %d: Key %d still found in g_Busy_Set after removal.\n", tid, test_key);
      } else {
          printf("Thread %d: Key %d NOT found in g_Busy_Set after removal.\n", tid, test_key);
      }
  }
__global__ void calcExchangePairs(int sector_id,int arr_size
){

  // printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
  //   MoRe_Hash->capacity,MoMo_Hash->capacity,
  //   Re_Hash->capacity,V_Hash->capacity,
  //   g_ghost_Hash->capacity,g_surface_Hash->capacity,g_Busy_Set->capacity
  // );
  if( blockIdx.x == 0 && threadIdx.x==0)
    printHashTableKernel(g_MoMo_Hash,200,0);
  __syncthreads();
  _type_lattice_count tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= arr_size)
    return;
  // if(tid >= 1)
  //   return;
  
  _type_lattice_id id = g_Pair_Atoms[tid];
  LatticeTypes::lat_type cur_type = getType(id);
  _type_lattice_size x = id%g_meta->size_x;
  _type_lattice_size y = (id / g_meta->size_x) % g_meta->size_y;
  _type_lattice_size z = id / (g_meta->size_x * g_meta->size_y);

  //if not in this domain then skip
  if(!(2 * g_meta->regions[sector_id * 6 + 0] <= x && x < g_meta->regions[sector_id * 6 + 1] && g_meta->regions[sector_id * 6 + 2] <= y && y < g_meta->regions[sector_id * 6 + 3] && g_meta->regions[sector_id * 6 + 4] <= z && z < g_meta->regions[sector_id * 6 + 5])){
    // 定义区域边界
    // const long int x_min = g_meta->regions[sector_id * 6 + 0];
    // const long int x_max = g_meta->regions[sector_id * 6 + 1];
    // const long int y_min = g_meta->regions[sector_id * 6 + 2];
    // const long int y_max = g_meta->regions[sector_id * 6 + 3];
    // const long int z_min = g_meta->regions[sector_id * 6 + 4];
    // const long int z_max = g_meta->regions[sector_id * 6 + 5];

    // // 检查每个维度的边界条件
    // bool x_ok = (2 * x_min <= x && x < x_max);
    // bool y_ok = (y_min <= y && y < y_max);
    // bool z_ok = (z_min <= z && z < z_max);

    // // 输出调试信息
    // printf("Sector %d: x=[%ld,%ld), y=[%ld,%ld), z=[%ld,%ld)\n", 
    //       sector_id, x_min, x_max, y_min, y_max, z_min, z_max);
    // printf("Point (%ld,%ld,%ld): x_ok=%d, y_ok=%d, z_ok=%d\n", 
    //       x, y, z, x_ok, y_ok, z_ok);
    return;
  }
    



  
  int numSIA;
  int numRe;
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
  while (2 * g_meta->regions[sector_id * 6 + 0] <= x && x < g_meta->regions[sector_id * 6 + 1] && g_meta->regions[sector_id * 6 + 2] <= y && y < g_meta->regions[sector_id * 6 + 3] && g_meta->regions[sector_id * 6 + 4] <= z && z < g_meta->regions[sector_id * 6 + 5]) 
  {
    loop_index++;

    // printf("busy set:");
    // for(int i=0;i<count;i++){
    //   if(g_Busy_Set->keys[i]!= -1 && g_Busy_Set->keys[i]!= -2)
    //     printf("(%d,%d),%d\t",tid,loop_index,g_Busy_Set->keys[i]);
    // }
    // printf("\n");
    if(tid == -1)
      printf("%d:%d (%d,%d,%d)\t",tid,loop_index,x,y,z);
    id = getId(x,y,z);
    // printf("(%d,%d) current id : (%d,%d)\n",tid,loop_index,id,getType(id));
    if(cur_type != getType(id)){
      printf("%d:%d id has wrong type %d - %d\n",tid,id,cur_type,getType(id));
      // abort();
    }

    if(hash_set_contains(g_Busy_Set,id)){
      continue;
    }else{
      ret = hash_set_insert(g_Busy_Set,id);
      busy++;
    }
    numSIA = 0;

    get1nn(x,y,z,nn_list);
    for(int i=0;i < LatticesList::MAX_1NN;i++){
      if(nn_list[i].type == LatticeTypes::MoMo || nn_list[i].type == LatticeTypes::MoRe)numSIA++;
    }

    if(numSIA >= 2) {
      ret = hash_set_remove(g_Busy_Set,id);
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
          ret = hash_set_remove(g_Busy_Set,id);
          if(ret == 1)
          busy--;
          return;
        }
    }

    int sum = tid + id + loop_index;
    int randomNumber = (sum % 14 + 14) % 14;  // 确保结果在0-13之间

    if(randomNumber < 0 || randomNumber >= 14){
        printf("%d : random num generate error %d\n",tid,randomNumber);
      }
    getRandomLattice(x,y,z,temp_x,temp_y,temp_z,randomNumber);
    randomLattice.id = getId(temp_x,temp_y,temp_z);


    if(hash_set_contains(g_Busy_Set,randomLattice.id)){
      ret = hash_set_remove(g_Busy_Set,id);
      if(ret == 1)
      busy--;
      continue;
    }else{
      ret = hash_set_insert(g_Busy_Set,randomLattice.id);
      busy++;
    }
    // printf("%d:is finish select random number\n",tid);
    randomLattice.type = getType(randomLattice.id);
    LatticeTypes::lat_type rantype = randomLattice.type;
    // printf("(%d,%d) (%d,%d) \n",tid,loop_index,randomLattice.id,getType(randomLattice.id));

    if(randomLattice.type == LatticeTypes::Mo || randomLattice.type == LatticeTypes::Re){
      if(cur_type == LatticeTypes::MoMo){
         if (randomLattice.type == LatticeTypes::Mo) { // 交换位置

          ret = hash_set_remove(g_MoMo_Hash,id);
          num_momo--;
          ret = hash_set_insert(g_MoMo_Hash,randomLattice.id);
          num_momo++;

          // printf("(%d,%d) (%d->%d)  , (%d->%d)\n",tid,loop_index,id,randomLattice.id,getType(id),getType(randomLattice.id));
        }
        else if(randomLattice.type == LatticeTypes::Re) { // Re (momo 变成 mo，re 变成 more)
          ret = hash_set_remove(g_Re_Hash,randomLattice.id);
          ret = hash_set_remove(g_MoMo_Hash,id);
          num_momo--;
          ret = hash_set_insert(g_MoRe_Hash,randomLattice.id);
          num_more++;
          cur_type = LatticeTypes::MoRe;
          // printf("(%d,%d) (%d->%d)  , (%d->%d)\n",tid,loop_index,id,randomLattice.id,getType(id),getType(randomLattice.id));
        }
      //   printf("1 current type %d - %d \t %d-%d\n",getId(temp_x,temp_y,temp_z),getType(getId(temp_x,temp_y,temp_z))
      //                                             ,randomLattice.id,randomLattice.type                                                           
      // );
      // printf("(%d,%d),(%d->%d),(%d->%d)\n",tid,loop_index,id,randomLattice.id,getType(id),getType(randomLattice.id));
        x = temp_x;
        y = temp_y;
        z = temp_z;
        
        if (isGhostLat(randomLattice.id)) {
          ret = hash_set_insert(g_ghost_Hash,randomLattice.id);
        }
        if (isSurfaceLat(randomLattice.id)){
          ret = hash_set_insert(g_surface_Hash,randomLattice.id);
        }
          
        if (isSurfaceLat(id)){
          ret = hash_set_insert(g_surface_Hash,id);
        }
      }
      else if(cur_type == LatticeTypes::MoRe){
        if(numRe == 0){
          if (randomLattice.type == LatticeTypes::Mo) { // 交换位置
            ret = hash_set_remove(g_MoRe_Hash,id);
            num_more--;
            ret  = hash_set_insert(g_MoRe_Hash,randomLattice.id);
            num_more++;
            // printf("(%d,%d) (%d->%d)  , (%d->%d)\n",tid,loop_index,id,randomLattice.id,getType(id),getType(randomLattice.id));
          } else { // Re 也是交换位置
            ret = hash_set_remove(g_Re_Hash,randomLattice.id);
            ret = hash_set_remove(g_MoRe_Hash,id);
            num_more--;
            // printf("%d:%d 2more removed:  %d ret = %d\n",tid,num_more,randomLattice.id,ret);
            ret = hash_set_insert(g_Re_Hash,id);
            ret = hash_set_insert(g_MoRe_Hash,randomLattice.id);
            // printf("(%d,%d) (%d->%d)  , (%d->%d)\n",tid,loop_index,id,randomLattice.id,getType(id),getType(randomLattice.id));
            num_more++;;
          }
          x = temp_x;
          y = temp_y;
          z = temp_z;
          // printf("2 current type %d\n",getType(getId(temp_x,temp_y,temp_z)));

          //TODO:
          if (isGhostLat(randomLattice.id)) {
            ret = hash_set_insert(g_ghost_Hash,randomLattice.id);
          }
          if (isSurfaceLat(randomLattice.id))
            ret = hash_set_insert(g_surface_Hash,randomLattice.id);
          if (isSurfaceLat(id)) 
            ret = hash_set_insert(g_surface_Hash,id);
        }
        else if(numRe > 0 && numRe < LatticesList::MAX_1NN){
          double ranmov = randomFloat(tid + id + loop_index);
          if(ranmov > (static_cast<double>(numRe) / LatticesList::MAX_1NN)) {
          if (randomLattice.type == LatticeTypes::Mo) {  // 交换位置
            ret = hash_set_remove(g_MoRe_Hash,id);
            num_more--;

            ret = hash_set_insert(g_MoRe_Hash,randomLattice.id);
            num_more++;
            // printf("(%d,%d) (%d->%d)  , (%d->%d)\n",tid,loop_index,id,randomLattice.id,getType(id),getType(randomLattice.id));
          } else if(randomLattice.type == LatticeTypes::Re) { // Re 也是交换位置
            ret = hash_set_remove(g_Re_Hash,randomLattice.id);
            ret = hash_set_remove(g_MoRe_Hash,id);
            num_more--;
            ret = hash_set_insert(g_Re_Hash,id);
            ret = hash_set_insert(g_MoRe_Hash,randomLattice.id);
            num_more++;
            // printf("(%d,%d) (%d->%d)  , (%d->%d)\n",tid,loop_index,id,randomLattice.id,getType(id),getType(randomLattice.id));
          }
          x = temp_x;
          y = temp_y;
          z = temp_z;
          // printf("3 current type %d\n",getType(getId(temp_x,temp_y,temp_z)));
          //TODO
          if (isGhostLat(randomLattice.id)) {
            ret = hash_set_insert(g_ghost_Hash,randomLattice.id);
          }
          if (isSurfaceLat(randomLattice.id)) 
            ret = hash_set_insert(g_surface_Hash,randomLattice.id);
          if (isSurfaceLat(id)) 
            ret = hash_set_insert(g_surface_Hash,id);
        }
      }else{
        ret = hash_set_remove(g_Busy_Set,randomLattice.id);
        if(ret == 1)
        busy--;
        ret = hash_set_remove(g_Busy_Set,id);
        if(ret == 1)
        busy--;
        // printf("%d : more : %d momo : %d \n",tid,num_more,num_momo);
        return;
      } 
    }else {
      ret = hash_set_remove(g_Busy_Set,randomLattice.id);
      if(ret == 1)
        busy--;
      ret = hash_set_remove(g_Busy_Set,id);
      if(ret == 1)
        busy--;
      continue;
    }
  }  
    ret = hash_set_remove(g_Busy_Set,randomLattice.id);
    if(ret == 1)
    busy--;
    ret = hash_set_remove(g_Busy_Set,id);
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


void recb_solver_GPU(
                    const unsigned int& sector_id,long int arr_size,
                    long int *&h_ghost_Hash,long int *&h_surface_Hash,
                    int *sizes
                  ){
    using namespace gpuSimulate;
    //在这里调用一个单核核函数对hash_set进行更新
  
    // printf("starting recb solver with GPU sector_id:%d,arr_size:%d\n",sector_id,arr_size);
    blockSize = 128;
    numBlocks = arr_size / blockSize + 1;
    hipLaunchKernelGGL(calcExchangePairs, blockSize, numBlocks, 0, 0, 
                      sector_id,arr_size
                      );
    hipDeviceSynchronize();
    // std::cin.get();

    

    //这是计算后传回的部分
    //ghost和surface区需要传回

    HANDLE_HIP(hipMemcpy(h_ghost_Hash, ghost_Hash->device_ptr()->keys, ghost_Hash->device_ptr()->capacity*sizeof(long int), hipMemcpyDeviceToHost));
    HANDLE_HIP(hipMemcpy(h_surface_Hash, surface_Hash->device_ptr()->keys, surface_Hash->device_ptr()->capacity*sizeof(long int), hipMemcpyDeviceToHost));
    ghost_Hash->clear();
    surface_Hash->clear();
    sizes[0]= ghost_Hash->device_ptr()->capacity;
    sizes[1]= surface_Hash->device_ptr()->capacity;

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

void initialize_gpu(int process_rank){
  int gpu_count;
  HANDLE_HIP(hipGetDeviceCount(&gpu_count));
  HANDLE_HIP(hipSetDevice(process_rank % gpu_count));
  HANDLE_HIP(hipDeviceReset());
}

void gpu_prepare(double& v, double& T, LatticesList *p_list, comm::ColoredDomain *_p_domain,
                std::vector<long int> pair_atoms
                ){
  using namespace gpuSimulate;

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
    std::cout << "Region " << index << ":\n"
              << "  X: [" << region_sector.x_low << ", " << region_sector.x_high << ")\n"
              << "  Y: [" << region_sector.y_low << ", " << region_sector.y_high << ")\n"
              << "  Z: [" << region_sector.z_low << ", " << region_sector.z_high << ")\n"
              << "  Size: (" 
              << (region_sector.x_high - region_sector.x_low) << ", "
              << (region_sector.y_high - region_sector.y_low) << ", "
              << (region_sector.z_high - region_sector.z_low) << ")\n";

    index++;
  }
  index = 0;
  printf("init region msg \n");
  HANDLE_HIP(hipMalloc((void**)&d_sum_rates, sizeof(double)));
  HANDLE_HIP(hipMalloc((void**)&d_event, sizeof(dev_event)));
  
  vacHostArrayCapacity = lattice_list->vac_hash.size() / 6 + 1;
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

  printf("init vac finished \n");
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
  hipHostMalloc((void**)&Pair_Atoms,pair_atoms.size()*sizeof(long int));
  HANDLE_HIP(hipMalloc((void**)&d_Pair_Atoms,pair_atoms.size()*sizeof(long int)));
  //对环境变量以及哈希表进行初始化
  //每组的meta需要重新赋值
  printf("init gpu_info msg \n");
  lattice_list->initGpuInfo_exchange(pair_atoms,region_msg,
                                    0,meta,Pair_Atoms,
                                    MoRe_Hash,MoMo_Hash,Re_Hash,V_Hash,Busy_Set
                                    );
  printf("init gpu_info msg end\n");
  HANDLE_HIP(hipMemcpy(d_Pair_Atoms, Pair_Atoms,pair_atoms.size()*sizeof(long int), hipMemcpyHostToDevice));

  int ghost_size = (meta.regions[1] + meta.ghost_x - meta.regions[0])*(meta.regions[3] + meta.ghost_y - meta.regions[2]) * (meta.regions[5] + meta.ghost_z - meta.regions[4])
                    - (meta.regions[1] - meta.regions[0])*(meta.regions[3] - meta.regions[2]) * (meta.regions[5] - meta.regions[4])
                    + (meta.regions[1]-meta.regions[0])*meta.ghost_y*meta.ghost_z
                    + meta.ghost_x*(meta.regions[3]-meta.regions[2])*meta.ghost_z
                    + meta.ghost_x*(meta.regions[5]-meta.regions[4])*meta.ghost_y;
  int surface_size = (meta.regions[1] - meta.regions[0])*(meta.regions[3] - meta.regions[2]) * (meta.regions[5] - meta.regions[4])
                      - (meta.regions[1] - meta.ghost_x - meta.regions[0])*(meta.regions[3] - meta.ghost_y - meta.regions[2]) * (meta.regions[5] - meta.ghost_z - meta.regions[4]);

  surface_Hash = new HIPHashSet(surface_size);
  ghost_Hash = new HIPHashSet(ghost_size);
  //这是对meta的首次更新，以后每次都得更新
  HANDLE_HIP(hipMalloc((void**)&d_meta,sizeof(dev_meta)));
  HANDLE_HIP(hipMemcpy(d_meta, &meta,sizeof(dev_meta), hipMemcpyHostToDevice));
  printf("init global pointer\n");
  //set hash pointer to gpu global pointer
  // printDevMeta(meta);
  hipLaunchKernelGGL(setGlobalPointer, 1, 1, 0, 0, 
      d_meta,
      MoRe_Hash->device_ptr(),MoMo_Hash->device_ptr(),Re_Hash->device_ptr(),V_Hash->device_ptr(),Busy_Set->device_ptr(),
      surface_Hash->device_ptr(),ghost_Hash->device_ptr(),d_Pair_Atoms
  );

  hipDeviceSynchronize();
  // std::cin.get();
  // GPUHashSet* hostPtr;
  // hipMemcpyFromSymbol(&hostPtr, g_MoMo_Hash, sizeof(GPUHashSet*));
  // printf("Global pointer value after synchronization: %p\n", hostPtr);
  // hipLaunchKernelGGL(printpointer, 1, 1, 0, 0,   );
  // system("read -p 'Press Enter to continue...' var");


  //这些是对返回CPU的空间
    printf("malloc CPU hashset\n");
    // hipHostMalloc((void**)&lattice_list->h_MoRe_Hash,MoRe_Hash->device_ptr()->capacity*sizeof(long int));
    // hipHostMalloc((void**)&lattice_list->h_MoMo_Hash,MoMo_Hash->device_ptr()->capacity*sizeof(long int));
    // hipHostMalloc((void**)&lattice_list->h_Re_Hash,Re_Hash->device_ptr()->capacity*sizeof(long int));
    // hipHostMalloc((void**)&lattice_list->h_V_Hash,V_Hash->device_ptr()->capacity*sizeof(long int));

    hipHostMalloc((void**)&lattice_list->h_ghost_Hash,ghost_Hash->device_ptr()->capacity*sizeof(long int));
    hipHostMalloc((void**)&lattice_list->h_surface_Hash,surface_Hash->device_ptr()->capacity*sizeof(long int));

  printf("finish gpu prepare\n");
  hipHostFree( Pair_Atoms);
}

void gpu_final(){
  using namespace gpuSimulate;

  // 释放GPU内存
  HANDLE_HIP(hipFree(d_sum_rates));
  HANDLE_HIP(hipFree(d_event));

  HANDLE_HIP(hipFree(deviceVacancyInfo));
  HANDLE_HIP(hipFree(deviceNeighbourInfo));

  hipHostFree(h_event);
  hipHostFree(sum_rates);

  hipHostFree(hostVacancyInfo);
  hipHostFree(hostNeighbourInfo);


  //z这里是recb_solver_GPU的部分
  delete MoRe_Hash;
  delete MoMo_Hash;
  delete Re_Hash;
  delete V_Hash;

  hipFree(d_Pair_Atoms);


  // hipHostFree(Pair_Atoms);

  delete ghost_Hash;
  delete surface_Hash;

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
                                int64_t sub_box_lattice_size_0,int64_t sub_box_lattice_size_1,int64_t sub_box_lattice_size_2,
                                int64_t neighbour_local_sub_box_0,int64_t neighbour_local_sub_box_1,int64_t neighbour_local_sub_box_2
){
  int BCC_DBX = 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= receive_len)
    return;

  int64_t x,y,z;
  long int id;
  if(dimension == 0){
    for(int i=index;i<receive_len;i+=blockDim.x){
      if(sector_id % 2 == 0){
        d_buffer[index].x += BCC_DBX * sub_box_lattice_size_0;
      }else{
        d_buffer[index].x -= BCC_DBX * (sub_box_lattice_size_0 + (neighbour_local_sub_box_0 - sub_box_lattice_size_0));
      }
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      z = d_buffer[index].z;
      id = getId(x,y,z);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove(g_V_Hash,id);
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove(g_Re_Hash,id);
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove(g_MoMo_Hash,id);
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove(g_MoRe_Hash,id);
      }

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }

  }
  else if(dimension == 1){
    for(int i=index;i<receive_len;i+=blockDim.x){
      x = d_buffer[index].x;
      if(sector_id == 0 || sector_id == 1 || sector_id == 4 || sector_id == 5) {
        d_buffer[index].y += sub_box_lattice_size_1;
      } else {
        d_buffer[index].y -= sub_box_lattice_size_1 + (neighbour_local_sub_box_1 - sub_box_lattice_size_1);
      }
      y = d_buffer[index].y;
      z = d_buffer[index].z;

      id = getId(x,y,z);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove(g_V_Hash,id);
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove(g_Re_Hash,id);
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove(g_MoMo_Hash,id);
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove(g_MoRe_Hash,id);
      }

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }
  }
  else if(dimension == 2){
    for(int i=index;i<receive_len;i+=blockDim.x){
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      if(sector_id < 4) {
        d_buffer[index].z += sub_box_lattice_size_2;
      }else{
        d_buffer[index].z -= sub_box_lattice_size_2+ (neighbour_local_sub_box_2 - sub_box_lattice_size_2);
      }
      z = d_buffer[index].z;

      id = getId(x,y,z);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove(g_V_Hash,id);
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove(g_Re_Hash,id);
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove(g_MoMo_Hash,id);
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove(g_MoRe_Hash,id);
      }

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert(g_MoRe_Hash,id);
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
                                int64_t sub_box_lattice_size_0,int64_t sub_box_lattice_size_1,int64_t sub_box_lattice_size_2,
                                int64_t neighbour_local_sub_box_0,int64_t neighbour_local_sub_box_1,int64_t neighbour_local_sub_box_2
){
  int BCC_DBX = 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= receive_len)
    return;

  int64_t x,y,z;
  long int id;
  if(dimension == 0){
    for(int i=index;i<receive_len;i+=blockDim.x){
      if(next_id % 2 == 1){
        d_buffer[index].x += BCC_DBX * sub_box_lattice_size_0;
      }else{
        d_buffer[index].x -= BCC_DBX * (sub_box_lattice_size_0 + (neighbour_local_sub_box_0 - sub_box_lattice_size_0));
      }
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      z = d_buffer[index].z;
      id = getId(x,y,z);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove(g_V_Hash,id);
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove(g_Re_Hash,id);
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove(g_MoMo_Hash,id);
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove(g_MoRe_Hash,id);
      }

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }

  }
  else if(dimension == 1){
    for(int i=index;i<receive_len;i+=blockDim.x){
      x = d_buffer[index].x;
      if(next_id == 2 || next_id == 3 || next_id == 6 || next_id == 7) {
        d_buffer[index].y += sub_box_lattice_size_1;
      } else {
        d_buffer[index].y -= sub_box_lattice_size_1 + (neighbour_local_sub_box_1 - sub_box_lattice_size_1);
      }
      y = d_buffer[index].y;
      z = d_buffer[index].z;

      id = getId(x,y,z);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove(g_V_Hash,id);
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove(g_Re_Hash,id);
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove(g_MoMo_Hash,id);
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove(g_MoRe_Hash,id);
      }

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert(g_MoRe_Hash,id);
          break;
        case LatticeTypes::Mo:
          break;
        default:
          break;
      }
    }
  }
  else if(dimension == 2){
    for(int i=index;i<receive_len;i+=blockDim.x){
      x = d_buffer[index].x;
      y = d_buffer[index].y;
      if(next_id > 3 ) {
        d_buffer[index].z += sub_box_lattice_size_2;
      }else{
        d_buffer[index].z -= sub_box_lattice_size_2+ (neighbour_local_sub_box_2 - sub_box_lattice_size_2);
      }
      z = d_buffer[index].z;

      id = getId(x,y,z);

      if(hash_set_contains(g_V_Hash,id)){
        hash_set_remove(g_V_Hash,id);
      }
      else if(hash_set_contains(g_Re_Hash,id)){
        hash_set_remove(g_Re_Hash,id);
      }
      else if(hash_set_contains(g_MoMo_Hash,id)){
        hash_set_remove(g_MoMo_Hash,id);
      }
      else if(hash_set_contains(g_MoRe_Hash,id)){
        hash_set_remove(g_MoRe_Hash,id);
      }

      switch(d_buffer[index].type){
        case LatticeTypes::V:
          hash_set_insert(g_V_Hash,id);
          break;
        case LatticeTypes::Re:
          hash_set_insert(g_Re_Hash,id);
          break;
        case LatticeTypes::MoMo:
          hash_set_insert(g_MoMo_Hash,id);
          break;
        case LatticeTypes::MoRe:
          hash_set_insert(g_MoRe_Hash,id);
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
    ChangeLattice_GPU *h_buffer;
    hipHostMalloc((void**)&h_buffer,receive_len*sizeof(receive_len));
    init_ChangeLattice_GPU(buffer,h_buffer,receive_len);

    ChangeLattice_GPU *d_buffer;
    HANDLE_HIP(hipMalloc((void**)&d_buffer,sizeof(ChangeLattice_GPU)*receive_len));
    HANDLE_HIP(hipMemcpy(d_buffer,&d_buffer,sizeof(ChangeLattice_GPU)*receive_len,hipMemcpyHostToDevice));
    int blockSize = 128;
    int blockNum = (receive_len)/blockSize + 1;
    hipLaunchKernelGGL(update_hash_set, blockSize, blockNum, 0, 0, 
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

}


void transferBufferToGPU2(ChangeLattice *buffer, int receive_len, const int dimension,
                          const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int next_id){
    ChangeLattice_GPU *h_buffer;
    hipHostMalloc((void**)&h_buffer,receive_len*sizeof(receive_len));
    init_ChangeLattice_GPU(buffer,h_buffer,receive_len);

    ChangeLattice_GPU *d_buffer;
    HANDLE_HIP(hipMalloc((void**)&d_buffer,sizeof(ChangeLattice_GPU)*receive_len));
    HANDLE_HIP(hipMemcpy(d_buffer,&d_buffer,sizeof(ChangeLattice_GPU)*receive_len,hipMemcpyHostToDevice));
    int blockSize = 128;
    int blockNum = (receive_len)/blockSize + 1;
    hipLaunchKernelGGL(update_hash_set2, blockSize, blockNum, 0, 0, 
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

}