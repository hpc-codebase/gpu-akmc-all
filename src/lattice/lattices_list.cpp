//
// Created by zhaorunchun on 2018-12-06.
// Update by genshen on 2018-12-09.
//

#include "lattices_list.h"
#include "type_define.h"
#include "utils/macros.h"
#include <comm/domain/region.hpp>
#include <iostream>
#include <random>
#include <logs/logs.h>

LatticesList::LatticesList(const LatListMeta meta) : meta(meta) {
  // _lattices = new Lattice **[meta.size_z];
  // for (_type_lattice_size z = 0; z < meta.size_z; z++) {
  //   _lattices[z] = new Lattice *[meta.size_y];
  //   for (_type_lattice_size y = 0; y < meta.size_y; y++) {
  //     _lattices[z][y] = new Lattice[meta.size_x];
  //   }
  // }
  // // set id (including setting local ids for ghost area)
  // _type_lattice_id id = 0;
  // for (_type_lattice_size z = 0; z < meta.size_z; z++) {
  //   for (_type_lattice_size y = 0; y < meta.size_y; y++) {
  //     for (_type_lattice_size x = 0; x < meta.size_x; x++) {
  //       _lattices[z][y][x].id = id++;
  //     }
  //   }
  // }
}

LatticesList::~LatticesList() {
  // for (_type_lattice_size z = 0; z < meta.size_z; z++) {
  //   for (_type_lattice_size y = 0; y < meta.size_y; y++) {
  //     delete[] _lattices[z][y];
  //   }
  //   delete[] _lattices[z];
  // }
  // delete[] _lattices;
}

// void LatticesList::forAllLattices(const func_lattices_callback callback) {
//   for (_type_lattice_size z = 0; z < meta.size_z; z++) {
//     for (_type_lattice_size y = 0; y < meta.size_y; y++) {
//       for (_type_lattice_size x = 0; x < meta.size_x; x++) {
//         if (!callback(x, y, z, _lattices[z][y][x])) {
//           return;
//         }
//       }
//     }
//   }
// }

_type_neighbour_status LatticesList::get1nnBoundaryStatus(_type_lattice_coord x, _type_lattice_coord y,
                                                          _type_lattice_coord z) {
  _type_neighbour_status flag = 0xFF; // binary 1 for keeping, 0 for abandon
  if (x == 0) {
    flag &= 0xF0; // invalid first 4 index.
  }
  if (x + 1 == meta.size_x) {
    flag &= 0x0F; // invalid last 4 index.
  }
  if (x % 2 == 0) {
    if (y == 0) {
      flag &= 0xCC; // 0b11001100 = 0x0CC
    }
    if (z == 0) {
      flag &= 0xAA; // 0b10101010 = 0xAA
    }
  } else {
    if (y + 1 == meta.size_y) {
      flag &= 0x33; // 0b00110011 = 0x033
    }
    if (z + 1 == meta.size_z) {
      flag &= 0x55; // 0b01010101 = 0x55
    }
  }
  return flag;
}

_type_neighbour_status LatticesList::get2nnBoundaryStatus(_type_lattice_coord x, _type_lattice_coord y,
                                                          _type_lattice_coord z) {
  _type_neighbour_status status = 0x3F; // 0b00111111
  // the order is from x to z
  // (increase x from lower to higher; if x is the same, increase y from lower
  // to higher; if y is the same, increase z from lower to higher.)
  if (x < 2) {
    status &= 0x3E; // 0b 0011 1110
  }
  if (y == 0) {
    status &= 0x3D; // 0b 0011 1101
  }
  if (z == 0) {
    status &= 0x3B; // 0b 0011 1011
  }
  if (z + 1 == meta.size_z) {
    status &= 0x37; // 0b 0011 0111
  }
  if (y + 1 == meta.size_y) {
    status &= 0x2F; // 0b 0010 1111
  }
  if (x + 2 >= meta.size_x) {
    status &= 0x1F; // 0b 0001 1111
  }
  return status;
}

Lattice &LatticesList::getLat(_type_lattice_id lid) {
  _type_lattice_coord x, y, z;
  meta.getCoordByLId(lid, &x, &y, &z);
  return _lattices[z][y][x];
}

Lattice &LatticesList::getLatByGid(_type_lattice_id gid) {
  _type_lattice_coord lx = gid % meta.g_box_x - meta.g_base_x;
  gid = gid / meta.g_box_x;
  _type_lattice_coord ly = gid % meta.g_box_y - meta.g_base_y;
  _type_lattice_coord lz = gid / meta.g_box_y - meta.g_base_z;
  return _lattices[lz][ly][lx];
}

Lattice *LatticesList::walk(_type_lattice_id id, const _type_lattice_offset offset_x,
                            const _type_lattice_offset offset_y, const _type_lattice_offset offset_z) {
  // step1: xyz coordinate relative to local ghost boundary
  _type_lattice_coord sx = id % meta.size_x;
  id = id / meta.size_x;
  // use half lattice const coord.
  _type_lattice_coord sy = sx % 2 == 0 ? 2 * (id % meta.size_y) : 2 * (id % meta.size_y) + 1;
  _type_lattice_coord sz = sx % 2 == 0 ? 2 * (id / meta.size_y) : 2 * (id / meta.size_y) + 1;

  // step 2: add offset
  sx += offset_x;
  sy += offset_y;
  sz += offset_z;

  // if out of index
  if (sx < 0 || sy < 0 || sz < 0 || sx >= meta.size_x || sy >= 2 * meta.size_y || sz >= 2 * meta.size_z) {
    return nullptr;
  }
  // if sx is even, sy and sz should also be even.
  if (sx % 2 == 0 && sy % 2 == 0 && sz % 2 == 0) {
    return &_lattices[sz / 2][sy / 2][sx];
  }
  // if sx is odd, sy and sz should also be odd.
  if (sx % 2 == 1 && sy % 2 == 1 && sz % 2 == 1) {
    return &_lattices[sz / 2][sy / 2][sx];
  }
  return nullptr;
}

void LatticesList::initGpuInfo_exchange(std::vector<long int> pair_atoms,
                                       std::vector<long int>region_sector,
                                       const unsigned int& sector_id,dev_meta &Meta,
                                       long int *Pair_Atoms,
                                       HIPHashSet *&MoRe_Hash,HIPHashSet *&MoMo_Hash,
                                       HIPHashSet *&Re_Hash,HIPHashSet *&V_Hash,
                                       HIPHashSet *&Busy_Set
                                       ){
  Meta.size_x = meta.size_x;
  Meta.size_y = meta.size_y;
  Meta.size_z = meta.size_z;

  Meta.box_x = meta.box_x;
  Meta.box_y = meta.box_y;
  Meta.box_z = meta.box_z;

  Meta.g_box_x = meta.g_box_x;
  Meta.g_box_y = meta.g_box_y;
  Meta.g_box_z = meta.g_box_z;

  Meta.g_base_x = meta.g_base_x;
  Meta.g_base_y = meta.g_base_y;
  Meta.g_base_z = meta.g_base_z;

  Meta.ghost_x = meta.ghost_x;
  Meta.ghost_y = meta.ghost_y;
  Meta.ghost_z = meta.ghost_z;

  Meta._max_id = meta._max_id;

  Meta.local_base_id = meta.local_base_id;

  for(int i=0;i<region_sector.size();i++){
    Meta.regions[i] = region_sector[i];
  }
 

  for(int i=0;i<pair_atoms.size();i++){
    Pair_Atoms[i] = pair_atoms[i];
  }

  std::unordered_set<_type_lattice_id> vac_set;//vac_hash是map其它的是set

  for (const auto& pair : vac_hash) {
        vac_set.insert(pair.first); // 提取键并插入集合
  }
  MoRe_Hash = new HIPHashSet(more_hash,more_hash.size()+momo_hash.size());
  MoMo_Hash = new HIPHashSet(momo_hash,more_hash.size()+momo_hash.size());
  Re_Hash = new HIPHashSet(re_hash);
  V_Hash = new HIPHashSet(vac_set);
  Busy_Set = new HIPHashSet(pair_atoms.size());

  this->MoRe_Hash = MoRe_Hash;
  this->MoMo_Hash = MoMo_Hash;
  this->Re_Hash = Re_Hash;
  this->V_Hash = V_Hash;
}


void LatticesList::initGpuInfo(const _type_lattice_count& total, _type_lattice_id *vac_idArray, 
                               dev_Vacancy *h_vacancy, dev_nnLattice *h_nnneighbour) {
  #pragma omp parallel for
  for(_type_lattice_count i = 0; i < total; i++){
    //int thread_id = omp_get_thread_num();
    //std::cout << " Thread " << thread_id << " is processing index " << i << std::endl;
    _type_lattice_id x = vac_idArray[i] % meta.size_x;
    _type_lattice_id y = (vac_idArray[i] / meta.size_x) % meta.size_y;
    _type_lattice_id z = vac_idArray[i] / (meta.size_x * meta.size_y);

    h_vacancy[i].id = vac_idArray[i];
    _type_lattice_count curTargetIndex = i * NN_TOTAL;
    if(x % 2 == 0){

      h_nnneighbour[curTargetIndex].id = getId(x - 1, y - 1, z - 1);
      h_nnneighbour[curTargetIndex].type._type = getType(h_nnneighbour[curTargetIndex].id); // 126 = 8 + 6 + 64 + 48

      h_nnneighbour[curTargetIndex + 1].id = getId(x - 1, y - 1, z);
      h_nnneighbour[curTargetIndex + 1].type._type = getType(h_nnneighbour[curTargetIndex + 1].id);

      h_nnneighbour[curTargetIndex + 2].id = getId(x - 1, y, z - 1);
      h_nnneighbour[curTargetIndex + 2].type._type = getType(h_nnneighbour[curTargetIndex + 2].id);

      h_nnneighbour[curTargetIndex + 3].id = getId(x - 1, y, z);
      h_nnneighbour[curTargetIndex + 3].type._type = getType(h_nnneighbour[curTargetIndex + 3].id);

      h_nnneighbour[curTargetIndex + 4].id = getId(x + 1, y - 1, z - 1);
      h_nnneighbour[curTargetIndex + 4].type._type = getType(h_nnneighbour[curTargetIndex + 4].id);

      h_nnneighbour[curTargetIndex + 5].id = getId(x + 1, y - 1, z);
      h_nnneighbour[curTargetIndex + 5].type._type = getType(h_nnneighbour[curTargetIndex + 5].id);

      h_nnneighbour[curTargetIndex + 6].id = getId(x + 1, y, z - 1);
      h_nnneighbour[curTargetIndex + 6].type._type = getType(h_nnneighbour[curTargetIndex + 6].id);

      h_nnneighbour[curTargetIndex + 7].id = getId(x + 1, y, z);
      h_nnneighbour[curTargetIndex + 7].type._type = getType(h_nnneighbour[curTargetIndex + 7].id);

    }else{
        
      h_nnneighbour[curTargetIndex].id = getId(x - 1, y, z);
      h_nnneighbour[curTargetIndex].type._type = getType(h_nnneighbour[curTargetIndex].id);

      h_nnneighbour[curTargetIndex + 1].id = getId(x - 1, y, z + 1);
      h_nnneighbour[curTargetIndex + 1].type._type = getType(h_nnneighbour[curTargetIndex + 1].id);

      h_nnneighbour[curTargetIndex + 2].id = getId(x - 1, y + 1, z);
      h_nnneighbour[curTargetIndex + 2].type._type = getType(h_nnneighbour[curTargetIndex + 2].id);

      h_nnneighbour[curTargetIndex + 3].id = getId(x - 1, y + 1, z + 1);
      h_nnneighbour[curTargetIndex + 3].type._type = getType(h_nnneighbour[curTargetIndex + 3].id);

      h_nnneighbour[curTargetIndex + 4].id = getId(x + 1, y, z);
      h_nnneighbour[curTargetIndex + 4].type._type = getType(h_nnneighbour[curTargetIndex + 4].id);

      h_nnneighbour[curTargetIndex + 5].id = getId(x + 1, y, z + 1);
      h_nnneighbour[curTargetIndex + 5].type._type = getType(h_nnneighbour[curTargetIndex + 5].id);

      h_nnneighbour[curTargetIndex + 6].id = getId(x + 1, y + 1, z);
      h_nnneighbour[curTargetIndex + 6].type._type = getType(h_nnneighbour[curTargetIndex + 6].id);

      h_nnneighbour[curTargetIndex + 7].id = getId(x + 1, y + 1, z + 1);
      h_nnneighbour[curTargetIndex + 7].type._type = getType(h_nnneighbour[curTargetIndex + 7].id);

    }

    h_nnneighbour[curTargetIndex + 8].id = getId(x - 2, y, z);
    h_nnneighbour[curTargetIndex + 8].type._type = getType(h_nnneighbour[curTargetIndex + 8].id);

    h_nnneighbour[curTargetIndex + 8 + 1].id = getId(x, y - 1, z);
    h_nnneighbour[curTargetIndex + 8 + 1].type._type = getType(h_nnneighbour[curTargetIndex + 8 + 1].id);

    h_nnneighbour[curTargetIndex + 8 + 2].id = getId(x, y, z - 1);
    h_nnneighbour[curTargetIndex + 8 + 2].type._type = getType(h_nnneighbour[curTargetIndex + 8 + 2].id);

    h_nnneighbour[curTargetIndex + 8 + 3].id = getId(x, y, z + 1);
    h_nnneighbour[curTargetIndex + 8 + 3].type._type = getType(h_nnneighbour[curTargetIndex + 8 + 3].id);

    h_nnneighbour[curTargetIndex + 8 + 4].id = getId(x, y + 1, z);
    h_nnneighbour[curTargetIndex + 8 + 4].type._type = getType(h_nnneighbour[curTargetIndex + 8 + 4].id);

    h_nnneighbour[curTargetIndex + 8 + 5].id = getId(x + 2, y, z);
    h_nnneighbour[curTargetIndex + 8 + 5].type._type = getType(h_nnneighbour[curTargetIndex + 8 + 5].id);

    for(int b = 0; b < 8; b++){
      _type_lattice_id x2 = h_nnneighbour[curTargetIndex + b].id % meta.size_x;
      _type_lattice_id y2 = (h_nnneighbour[curTargetIndex + b].id / meta.size_x) % meta.size_y;
      _type_lattice_id z2 = h_nnneighbour[curTargetIndex + b].id / (meta.size_x * meta.size_y);

      if(x2 % 2 == 0){
          
        h_nnneighbour[curTargetIndex + 14 + b * 8].id = getId(x2 - 1, y2 - 1, z2 - 1);
        h_nnneighbour[curTargetIndex + 14 + b * 8].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8].id); // 14 = 6 + 8

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 1].id = getId(x2 - 1, y2 - 1, z2);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 1].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 1].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 2].id = getId(x2 - 1, y2, z2 - 1);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 2].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 2].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 3].id = getId(x2 - 1, y2, z2);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 3].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 3].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 4].id = getId(x2 + 1, y2 - 1, z2 - 1);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 4].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 4].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 5].id = getId(x2 + 1, y2 - 1, z2);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 5].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 5].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 6].id = getId(x2 + 1, y2, z2 - 1);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 6].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 6].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 7].id = getId(x2 + 1, y2, z2);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 7].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 7].id);

      }else{

        h_nnneighbour[curTargetIndex + 14 + b * 8].id = getId(x2 - 1, y2, z2);
        h_nnneighbour[curTargetIndex + 14 + b * 8].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 1].id = getId(x2 - 1, y2, z2 + 1);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 1].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 1].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 2].id = getId(x2 - 1, y2 + 1, z2);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 2].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 2].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 3].id = getId(x2 - 1, y2 + 1, z2 + 1);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 3].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 3].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 4].id = getId(x2 + 1, y2, z2);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 4].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 4].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 5].id = getId(x2 + 1, y2, z2 + 1);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 5].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 5].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 6].id = getId(x2 + 1, y2 + 1, z2);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 6].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 6].id);

        h_nnneighbour[curTargetIndex + 14 + b * 8 + 7].id = getId(x2 + 1, y2 + 1, z2 + 1);
        h_nnneighbour[curTargetIndex + 14 + b * 8 + 7].type._type = getType(h_nnneighbour[curTargetIndex + 14 + b * 8 + 7].id);

      }

      h_nnneighbour[curTargetIndex + 78 + b * 6].id = getId(x2 - 2, y2, z2);
      h_nnneighbour[curTargetIndex + 78 + b * 6].type._type = getType(h_nnneighbour[curTargetIndex + 78 + b * 6].id); // 78 = 64 + 14

      h_nnneighbour[curTargetIndex + 78 + b * 6 + 1].id = getId(x2, y2 - 1, z2);
      h_nnneighbour[curTargetIndex + 78 + b * 6 + 1].type._type = getType(h_nnneighbour[curTargetIndex + 78 + b * 6 + 1].id);

      h_nnneighbour[curTargetIndex + 78 + b * 6 + 2].id = getId(x2, y2, z2 - 1);
      h_nnneighbour[curTargetIndex + 78 + b * 6 + 2].type._type = getType(h_nnneighbour[curTargetIndex + 78 + b * 6 + 2].id);

      h_nnneighbour[curTargetIndex + 78 + b * 6 + 3].id = getId(x2, y2, z2 + 1);
      h_nnneighbour[curTargetIndex + 78 + b * 6 + 3].type._type = getType(h_nnneighbour[curTargetIndex + 78 + b * 6 + 3].id);

      h_nnneighbour[curTargetIndex + 78 + b * 6 + 4].id = getId(x2, y2 + 1, z2);
      h_nnneighbour[curTargetIndex + 78 + b * 6 + 4].type._type = getType(h_nnneighbour[curTargetIndex + 78 + b * 6 + 4].id);

      h_nnneighbour[curTargetIndex + 78 + b * 6 + 5].id = getId(x2 + 2, y2, z2);
      h_nnneighbour[curTargetIndex + 78 + b * 6 + 5].type._type = getType(h_nnneighbour[curTargetIndex + 78 + b * 6 + 5].id);

    }
  }
}
void LatticesList::update_hash_CPU(){
    more_hash.clear();
    momo_hash.clear();
    re_hash.clear();
    vac_hash.clear();

    this->MoRe_Hash->copyToHost(more_hash);
    this->MoMo_Hash->copyToHost(momo_hash);
    this->Re_Hash->copyToHost(re_hash);

    

    std::unordered_set<_type_lattice_id> tmp_vac_set;
    this->V_Hash->copyToHost(tmp_vac_set);

    for (const auto &key : tmp_vac_set) {
        vac_hash[key] = VacancyHash{};
    }
  
}
void LatticesList::updateGpuInfo(_type_lattice_id& to_x, _type_lattice_id& to_y, _type_lattice_id& to_z, dev_nnLattice *h_nnneighbour_temp) {

  if(to_x % 2 == 0){
    h_nnneighbour_temp[0].id = getId(to_x - 1, to_y - 1, to_z - 1);
    h_nnneighbour_temp[0].type._type = getType(h_nnneighbour_temp[0].id);

    h_nnneighbour_temp[1].id = getId(to_x - 1, to_y - 1, to_z);
    h_nnneighbour_temp[1].type._type = getType(h_nnneighbour_temp[1].id);

    h_nnneighbour_temp[2].id = getId(to_x - 1, to_y, to_z - 1);
    h_nnneighbour_temp[2].type._type = getType(h_nnneighbour_temp[2].id);

    h_nnneighbour_temp[3].id = getId(to_x - 1, to_y, to_z);
    h_nnneighbour_temp[3].type._type = getType(h_nnneighbour_temp[3].id);

    h_nnneighbour_temp[4].id = getId(to_x + 1, to_y - 1, to_z - 1);
    h_nnneighbour_temp[4].type._type = getType(h_nnneighbour_temp[4].id);

    h_nnneighbour_temp[5].id = getId(to_x + 1, to_y - 1, to_z);
    h_nnneighbour_temp[5].type._type = getType(h_nnneighbour_temp[5].id);

    h_nnneighbour_temp[6].id = getId(to_x + 1, to_y, to_z - 1);
    h_nnneighbour_temp[6].type._type = getType(h_nnneighbour_temp[6].id);

    h_nnneighbour_temp[7].id = getId(to_x + 1, to_y, to_z);
    h_nnneighbour_temp[7].type._type = getType(h_nnneighbour_temp[7].id);

  }else{

    h_nnneighbour_temp[0].id = getId(to_x - 1, to_y, to_z);
    h_nnneighbour_temp[0].type._type = getType(h_nnneighbour_temp[0].id);

    h_nnneighbour_temp[1].id = getId(to_x - 1, to_y, to_z + 1);
    h_nnneighbour_temp[1].type._type = getType(h_nnneighbour_temp[1].id);

    h_nnneighbour_temp[2].id = getId(to_x - 1, to_y + 1, to_z);
    h_nnneighbour_temp[2].type._type = getType(h_nnneighbour_temp[2].id);

    h_nnneighbour_temp[3].id = getId(to_x - 1, to_y + 1, to_z + 1);
    h_nnneighbour_temp[3].type._type = getType(h_nnneighbour_temp[3].id);

    h_nnneighbour_temp[4].id = getId(to_x + 1, to_y, to_z);
    h_nnneighbour_temp[4].type._type = getType(h_nnneighbour_temp[4].id);

    h_nnneighbour_temp[5].id = getId(to_x + 1, to_y, to_z + 1);
    h_nnneighbour_temp[5].type._type = getType(h_nnneighbour_temp[5].id);

    h_nnneighbour_temp[6].id = getId(to_x + 1, to_y + 1, to_z);
    h_nnneighbour_temp[6].type._type = getType(h_nnneighbour_temp[6].id);

    h_nnneighbour_temp[7].id = getId(to_x + 1, to_y + 1, to_z + 1);
    h_nnneighbour_temp[7].type._type = getType(h_nnneighbour_temp[7].id);

  }

  h_nnneighbour_temp[8].id = getId(to_x - 2, to_y, to_z);
  h_nnneighbour_temp[8].type._type = getType(h_nnneighbour_temp[8].id);

  h_nnneighbour_temp[9].id = getId(to_x, to_y - 1, to_z);
  h_nnneighbour_temp[9].type._type = getType(h_nnneighbour_temp[9].id);

  h_nnneighbour_temp[10].id = getId(to_x, to_y, to_z - 1);
  h_nnneighbour_temp[10].type._type = getType(h_nnneighbour_temp[10].id);

  h_nnneighbour_temp[11].id = getId(to_x, to_y, to_z + 1);
  h_nnneighbour_temp[11].type._type = getType(h_nnneighbour_temp[11].id);

  h_nnneighbour_temp[12].id = getId(to_x, to_y + 1, to_z);
  h_nnneighbour_temp[12].type._type = getType(h_nnneighbour_temp[12].id);

  h_nnneighbour_temp[13].id = getId(to_x + 2, to_y, to_z);
  h_nnneighbour_temp[13].type._type = getType(h_nnneighbour_temp[13].id);
        
  #pragma omp parallel for
  for(int b = 0; b < 8; b++){
    _type_lattice_id x4 = h_nnneighbour_temp[b].id % meta.size_x;
    _type_lattice_id y4 = (h_nnneighbour_temp[b].id / meta.size_x) % meta.size_y;
    _type_lattice_id z4 = h_nnneighbour_temp[b].id / (meta.size_x * meta.size_y);

    if(x4 % 2 == 0){
        
      h_nnneighbour_temp[14 + b * 8].id = getId(x4 - 1, y4 - 1, z4 - 1);
      h_nnneighbour_temp[14 + b * 8].type._type = getType(h_nnneighbour_temp[14 + b * 8].id);

      h_nnneighbour_temp[14 + b * 8 + 1].id = getId(x4 - 1, y4 - 1, z4);
      h_nnneighbour_temp[14 + b * 8 + 1].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 1].id);

      h_nnneighbour_temp[14 + b * 8 + 2].id = getId(x4 - 1, y4, z4 - 1);
      h_nnneighbour_temp[14 + b * 8 + 2].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 2].id);

      h_nnneighbour_temp[14 + b * 8 + 3].id = getId(x4 - 1, y4, z4);
      h_nnneighbour_temp[14 + b * 8 + 3].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 3].id);

      h_nnneighbour_temp[14 + b * 8 + 4].id = getId(x4 + 1, y4 - 1, z4 - 1);
      h_nnneighbour_temp[14 + b * 8 + 4].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 4].id);

      h_nnneighbour_temp[14 + b * 8 + 5].id = getId(x4 + 1, y4 - 1, z4);
      h_nnneighbour_temp[14 + b * 8 + 5].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 5].id);

      h_nnneighbour_temp[14 + b * 8 + 6].id = getId(x4 + 1, y4, z4 - 1);
      h_nnneighbour_temp[14 + b * 8 + 6].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 6].id);

      h_nnneighbour_temp[14 + b * 8 + 7].id = getId(x4 + 1, y4, z4);
      h_nnneighbour_temp[14 + b * 8 + 7].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 7].id);
          
    }else{

      h_nnneighbour_temp[14 + b * 8].id = getId(x4 - 1, y4, z4);
      h_nnneighbour_temp[14 + b * 8].type._type = getType(h_nnneighbour_temp[14 + b * 8].id);

      h_nnneighbour_temp[14 + b * 8 + 1].id = getId(x4 - 1, y4, z4 + 1);
      h_nnneighbour_temp[14 + b * 8 + 1].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 1].id);

      h_nnneighbour_temp[14 + b * 8 + 2].id = getId(x4 - 1, y4 + 1, z4);
      h_nnneighbour_temp[14 + b * 8 + 2].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 2].id);

      h_nnneighbour_temp[14 + b * 8 + 3].id = getId(x4 - 1, y4 + 1, z4 + 1);
      h_nnneighbour_temp[14 + b * 8 + 3].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 3].id);

      h_nnneighbour_temp[14 + b * 8 + 4].id = getId(x4 + 1, y4, z4);
      h_nnneighbour_temp[14 + b * 8 + 4].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 4].id);

      h_nnneighbour_temp[14 + b * 8 + 5].id = getId(x4 + 1, y4, z4 + 1);
      h_nnneighbour_temp[14 + b * 8 + 5].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 5].id);

      h_nnneighbour_temp[14 + b * 8 + 6].id = getId(x4 + 1, y4 + 1, z4);
      h_nnneighbour_temp[14 + b * 8 + 6].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 6].id);

      h_nnneighbour_temp[14 + b * 8 + 7].id = getId(x4 + 1, y4 + 1, z4 + 1);
      h_nnneighbour_temp[14 + b * 8 + 7].type._type = getType(h_nnneighbour_temp[14 + b * 8 + 7].id);

    }

    h_nnneighbour_temp[78 + b * 6].id = getId(x4 - 2, y4, z4);
    h_nnneighbour_temp[78 + b * 6].type._type = getType(h_nnneighbour_temp[78 + b * 6].id);

    h_nnneighbour_temp[78 + b * 6 + 1].id = getId(x4, y4 - 1, z4);
    h_nnneighbour_temp[78 + b * 6 + 1].type._type = getType(h_nnneighbour_temp[78 + b * 6 + 1].id);

    h_nnneighbour_temp[78 + b * 6 + 2].id = getId(x4, y4, z4 - 1);
    h_nnneighbour_temp[78 + b * 6 + 2].type._type = getType(h_nnneighbour_temp[78 + b * 6 + 2].id);

    h_nnneighbour_temp[78 + b * 6 + 3].id = getId(x4, y4, z4 + 1);
    h_nnneighbour_temp[78 + b * 6 + 3].type._type = getType(h_nnneighbour_temp[78 + b * 6 + 3].id);

    h_nnneighbour_temp[78 + b * 6 + 4].id = getId(x4, y4 + 1, z4);
    h_nnneighbour_temp[78 + b * 6 + 4].type._type = getType(h_nnneighbour_temp[78 + b * 6 + 4].id);

    h_nnneighbour_temp[78 + b * 6 + 5].id = getId(x4 + 2, y4, z4);
    h_nnneighbour_temp[78 + b * 6 + 5].type._type = getType(h_nnneighbour_temp[78 + b * 6 + 5].id);

  }
}

LatticeTypes::lat_type LatticesList::getType(const _type_lattice_id& latti_id) {
  if      (vac_hash.count(latti_id))  return LatticeTypes::V;
  else if (re_hash.count(latti_id))   return LatticeTypes::Re;
  else if (mn_hash.count(latti_id))   return LatticeTypes::Mn;
  else if (ni_hash.count(latti_id))   return LatticeTypes::Ni;
  else if (si_hash.count(latti_id))   return LatticeTypes::Si;
  else if (momo_hash.count(latti_id)) return LatticeTypes::MoMo;
  else if (more_hash.count(latti_id)) return LatticeTypes::MoRe;
  else if (rere_hash.count(latti_id)) return LatticeTypes::ReRe;
  else                                return LatticeTypes::Mo;
}