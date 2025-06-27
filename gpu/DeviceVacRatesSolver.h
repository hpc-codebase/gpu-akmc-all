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
    ~HIPHashSet();

    // 返回设备端指针
    GPUHashSet* device_ptr() { return d_table; }
    
};

int init_ChangeLattice_GPU(ChangeLattice *buffer,ChangeLattice_GPU *h_buffer,int len);
#endif /* DEVICEVACRATESSOVLER_H */