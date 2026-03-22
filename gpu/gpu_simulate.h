#ifndef GPU_SIMULATE_H
#define GPU_SIMULATE_H

#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include "gpuError.h"
#include "gpu_comm.h"
#include "DeviceVacRatesSolver.h"
#include <cassert>
#include <iostream>
#include <hiprand/hiprand_kernel.h>

#include <logs/logs.h>
#include "../models/abvi/kmc.h"
#include "../models/abvi/box.h"
#include "../models/abvi/rate/vacancy_rates_solver.h"
#include "../models/abvi/defect/vacancy_list.h"
#include "../models/abvi/rate/bonds/pair_bond.hpp"
#include "../models/abvi/rate/rates_types.h"
#include "../models/abvi/event.h"

#include "../src/type_define.h"
#include "../src/lattice/lattice.h"
#include "../src/lattice/lattices_list.h"
#include "../src/lattice/lattice_types.h"
#include "../src/lattice/lattice_list_meta.h"

void recb_solver_GPU(
                    const unsigned int& sector_id,long int arr_size,
                    long int *&h_ghost_Hash,long int *&h_surface_Hash,
                    long int *&h_ghost_type, long int *&h_surface_type,
                    int *sizes
                  );
double calculate_GPU(_type_lattice_id *vac_idArray,  const _type_lattice_count& vac_count);
double calculate_GPU_region(int x_low, int x_high, int y_low, int y_high, int z_low, int z_high);

void selectAndPerformEventGPU(double excepted_rand, int rank, _type_lattice_count step, int sect,
                              std::array<std::unordered_set<_type_lattice_id>, 7>& exchange_ghost, const unsigned int sector_id,
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_x, 
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_y,
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_z,
                              std::array<comm::Region<comm::_type_lattice_size>, 56>& ghost_region);

void initialize_gpu(int process_rank);

void gpu_final();

void gpu_prepare(double& v, double& T, LatticesList *p_list, comm::ColoredDomain *_p_domain,
                std::vector<long int> pair_atoms
                );

void rccl_smoke_test();

void exchange_lattices(dev_event* h_event);

void add_surface(_type_lattice_id& surface_id,
                 std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_x, 
                 std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_y,
                 std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_z);
                 
void output_time();

__device__ LatticeTypes::lat_type getType(const long int& id);

void transferBufferToGPU(ChangeLattice *buffer, int receive_len, const int dimension,
  const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int id);
void transferBufferToGPU2(ChangeLattice *buffer, int receive_len, const int dimension,
  const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int next_id);

void transferBufferToGPU_Device(ChangeLattice_GPU *d_buffer, int receive_len, const int dimension,
  const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int id);
void transferBufferToGPU2_Device(ChangeLattice_GPU *d_buffer, int receive_len, const int dimension,
  const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int next_id);

void packChangeLatticeFromIds_Device(const _type_lattice_id *d_ids, ChangeLattice_GPU *d_out, int n);

struct CommBox {
  _type_lattice_id x_low;
  _type_lattice_id x_high;
  _type_lattice_id y_low;
  _type_lattice_id y_high;
  _type_lattice_id z_low;
  _type_lattice_id z_high;
};

void commSetGhostRegions(const comm::Region<comm::_type_lattice_size> *ghost_region, int n);
void commInitGhostBuckets(int capacity);
void commClearGhostBuckets();
void commBuildGhostBucketsFromList(int sector_id, const long int *d_ids, int n);
void commCompactGhostBucket(int bucket, long int *d_out, int *d_count);
int commGhostCapacity();

void commSetSurfaceBoxes(const CommBox *boxes, int boxes_count, const int *offsets, int offsets_count);
void commInitSurfaceBuckets(int capacity);
void commClearSurfaceBuckets();
void commBuildSurfaceBucketsFromList(const long int *d_ids, int n);
void commCompactSurfaceBucket(int bucket, long int *d_out, int *d_count);
int commSurfaceCapacity();

void commClearSurfaceSector(int sect);
void commInsertGhostId(int sector_id, long int id);
void commInsertSurfaceId(long int id);

void commForwardFromRecvPayload(const ChangeLattice_GPU *d_payload, int n, int sector_id);
// 线程事务状态机结构体
struct AtomTx {
    _type_lattice_id src_id;   // 原原子ID
    _type_lattice_id dst_id;   // 目标位置ID
    bool active;               // 是否仍在活跃（活没干完）
    // bool locked_src;           // 是否成功锁住自己
    // bool locked_dst;           // 是否成功锁住目标
    bool success;              // 本轮物理交换是否成功
    int op_type;               // 反应类型：1-移动，2-身份转换...
};
// __global__ void calcExchangePairs(dev_meta meta,long int *Pair_Atoms,
//                             HIPHashSet *MoRe_Hash,HIPHashSet *MoMo_Hash,
//                             HIPHashSet *Re_Hash,HIPHashSet *V_Hash,
//                             HIPHashSet *Busy_Set,
//                             HIPHashSet *ghost_Hash,HIPHashSet *surface_Hash
// );

// __device__ int isSurfaceLat(_type_lattice_id lid);

// __device__ int isGhostLat(_type_lattice_id lid);

// __device__ void getCoordByLId(_type_lattice_id lid, _type_lattice_coord *x, _type_lattice_coord *y,
//                             _type_lattice_coord *z);

// __device__ float randomFloat(unsigned long long seed);

// __device__ void getRandomLattice(const _type_lattice_coord& x, const _type_lattice_coord& y, const _type_lattice_coord& z,
//                                 _type_lattice_coord& temp_x, _type_lattice_coord& temp_y, _type_lattice_coord& temp_z, const int& randomValue){

// __device__ int randomInt(unsigned long long seed,int min,int max);

// __device__ void get2nn(_type_lattice_coord x, _type_lattice_coord y, _type_lattice_coord z,
//                               dev_nnLattice *_2nn_list);

// __device__ void get1nn(_type_lattice_coord x, _type_lattice_coord y, _type_lattice_coord z,
//                   dev_nnLattice *_1nn_list);

// __device__ void getnnlattice(_type_lattice_id latti_id, dev_nnLattice *_1nn_list_lattice);

// __device__ _type_lattice_id getId(_type_lattice_size x,_type_lattice_size y,_type_lattice_size z);

// __device__ LatticeTypes::lat_type getType(const _type_lattice_id& id);

// __global__ void setGlobalPointer(dev_meta meta,
//                             HIPHashSet *MoRe_Hash,HIPHashSet *MoMo_Hash,
//                             HIPHashSet *Re_Hash,HIPHashSet *V_Hash,HIPHashSet *Busy_Set);

// __device__ int hash_set_contains(const int* table, _type_lattice_id key);

// __device__ int hash_set_insert(int* table, _type_lattice_id key);

// __device__ int hash_set_remove(int* table, _type_lattice_id key);
#endif /*GPU_SIMULATE_H*/
