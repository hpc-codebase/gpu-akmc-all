//
// Created by genshen on 2018-12-12.
//

#include <cassert>
#include <omp.h>
#include "env.h"
#include "kmc.h"
#include "rate/itl_rates_solver.h"
#include "rate/vacancy_rates_solver.h"
#include "recombine.h"
#include "utils/random/random.h"
#include <iostream>
#include <logs/logs.h>
#include "../gpu/gpu_simulate.h"
#include "utils/simulation_domain.h"
#include "../../gpu/DeviceVacRatesSolver.h"
#include <map>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include "../../profiles/config/lattice_types_string.h"

ABVIModel::ABVIModel(Box *box, double v, double T) : box(box), v(v), T(T) {}

// 这个函数的主要作用是计算指定仿真区域内的某种类型的迁移速率（transition rates）。
// 在材料模拟或分子动力学中，迁移速率通常用于描述原子、分子或其他粒子之间的运动、迁移或反应速度。具体来说，这个函数执行以下操作：
// 初始化一些变量和对象，如ItlRatesSolver和VacRatesSolver，这些对象可能用于计算迁移速率。
// 使用嵌套循环遍历指定仿真区域内的所有格点。这些嵌套循环涵盖三个维度：x、y 和 z。
// 对于每个格点，检查格点上的晶格类型（Lattice type），可能是"dumbbell"或"vacancy"，分别代表"双原子晶格"和"空位"。根据不同的晶格类型，选择不同的处理路径。
// 对于"dumbbell"类型的晶格，执行一系列操作来计算与相邻格点之间的迁移速率。这包括获取与当前晶格相邻的格点信息、更新迁移速率等。
// 迁移速率计算似乎涉及到一个lambda函数，该函数返回从当前晶格到相邻晶格的迁移速率，并将其累加到sum_rates中。
// 对于"vacancy"类型的晶格，执行类似的操作，计算空位的迁移速率，并将其累加到sum_rates中。
// 最后，通过调用defectGenRate()函数，将生成的缺陷（defect）的生成速率添加到sum_rates中。
// 返回sum_rates，即在整个仿真区域内计算得到的总迁移速率。
// 总的来说，用于计算指定仿真区域内不同晶格类型的迁移速率，并将它们累加到一起以获得总迁移速率。

// _type_rate ABVIModel::calcRates_GPU(const comm::Region<comm::_type_lattice_size> region) {
//   return calcRatesGPU(region, box, v, T);
// }

_type_rate ABVIModel::calcRates(const comm::Region<comm::_type_lattice_size> region) {
  _type_rate sum_rates = 0;
  VacRatesSolver vac_rate(*(box->lattice_list), v, T);

  // for (const auto& it : box->lattice_list->vac_hash) {
  //   _type_lattice_size x = it.first % box->lattice_list->meta.size_x;
  //   _type_lattice_size y = (it.first / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
  //   _type_lattice_size z = it.first / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);
  //   if(2 * region.x_low <= x && x < 2 * region.x_high && region.y_low <= y && y < region.y_high && region.z_low <= z && z < region.z_high) {
  for (_type_lattice_id z = region.z_low; z < region.z_high; z++) {
    for (_type_lattice_id y = region.y_low; y < region.y_high; y++) {
      for (_type_lattice_id x = 2 * region.x_low; x < 2 * region.x_high; x++) {
        _type_lattice_id latti_id = box->lattice_list->getId(x, y, z);
        if (box->lattice_list->vac_hash.count(latti_id)) { // vacancy
        VacancyHash &vacancyhash = box->lattice_list->vac_hash.at(latti_id);
        Lattice lat_list[LatticesList::MAX_1NN]; // todo new array many times.
        box->lattice_list->get1nn2(x, y, z, lat_list);
        // kiwi::logs::v(" ", " latti_id is : {}. \n", latti_id);
        // for (int i = 0; i < 8; i++) {
        //   kiwi::logs::v(" ", " nn_index is : {} nn_id is : {} nn_type is : {} . \n", i, lat_list[i].id, lat_list[i].type._type);
        // }
        _type_neighbour_status nei_status = box->lattice_list->get1nnStatus(x, y, z);//获取空位状态
        vacancyhash.beforeRatesUpdate2(lat_list, nei_status);
        Lattice source_latti;
        source_latti.id = latti_id;
        source_latti.type = LatticeTypes{LatticeTypes::V};
        //kiwi::logs::v(" ", " avail_trans_dir is : {} .\n", vacancy.avail_trans_dir);
        vacancyhash.updateRates2(source_latti, lat_list, nei_status,
                            [&source_latti, &vac_rate, &sum_rates](Lattice *lat_nei,
                                                              const LatticeTypes::lat_type ghost_atom,
                                                              const _type_dir_id _1nn_offset) -> _type_rate {
                              _type_rate rate = vac_rate.rate(source_latti, *lat_nei, ghost_atom, _1nn_offset);
                              sum_rates += rate; // add this rate to sum
                              //kiwi::logs::v(" ", " rates is : {} .\n", rate);
                              return rate;
                            });
        }
      }
    }
  }
  sum_rates += defectGenRate();//0
  return sum_rates;
}

_type_rate ABVIModel::calcRatesGPU(const comm::Region<comm::_type_lattice_size> region, int sect) {
  _type_lattice_id *vac_idArray;
  vac_idArray = (_type_lattice_id *)malloc(box->lattice_list->vac_hash.size() * sizeof(_type_lattice_id));
  _type_lattice_count vac_count = 0;

  for (const auto& pair : box->lattice_list->vac_hash) {
    _type_lattice_size x = pair.first % box->lattice_list->meta.size_x;
    _type_lattice_size y = (pair.first / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
    _type_lattice_size z = pair.first / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);
    // 传输当前扇区剩下的空位
    if(2 * region.x_low <= x && x < 2 * region.x_high && region.y_low <= y && y < region.y_high && region.z_low <= z && z < region.z_high) {
      vac_idArray[vac_count++] = pair.first;
    }
  }
  return vac_count == 0? 0 : calculate_GPU(vac_idArray, vac_count);
}

_type_rate ABVIModel::defectGenRate() { return env::global_env.defect_gen_rate; }

void ABVIModel::selectAndPerformOnGPU(const _type_rate rate, int rank, _type_lattice_count step, int sect, const unsigned int sector_id, const unsigned int next_sector_id) {
  selectAndPerformEventGPU(rate, rank, step, sect, exchange_ghost, sector_id, exchange_surface_x, exchange_surface_y, exchange_surface_z, ghost_region);
}

// lat_region region：表示某个区域的参数。
// _type_rate excepted_rate：期望的速率。取值在[0, sum_rates)
// _type_rate sum_rates：总速率。
// 函数内部操作：
// rate_accumulator：初始化一个机率累加器，用于追踪已累加的机率。
// selected_event：创建一个 event::SelectedEvent 对象，用于存储所选事件的信息，初始事件类型为 event::DefectGen（缺陷生成）。
// 通过三个嵌套的循环遍历指定区域内的晶格：
// 外循环遍历 z 轴范围。
// 中循环遍历 y 轴范围。
// 内循环遍历 x 轴范围。
// 对于每个晶格，检查晶格的类型：
// 如果是 dumbbell 类型的晶格，进入相应的分支：
// 遍历 itl_ref 对象的机率数组，累加机率到 rate_accumulator 中。
// 如果 rate_accumulator 大于 excepted_rate，则表示找到了事件：
// 获取与 rate_index 相关的相邻晶格 _1nn_list。
// 计算事件类型为 event::DumbbellTrans，并设置相关的事件信息，包括起始晶格、目标晶格、目标标签和旋转方向。
// 使用 goto 标签 EVENT_FOUND 跳出循环。
// 如果是 vacancy 类型的晶格，进入相应的分支：
// 遍历 vacancy 对象的速率数组，累加速率到 rate_accumulator 中。
// 如果 rate_accumulator 大于 excepted_rate，则表示找到了事件：
// 获取与 rate_index 相关的相邻晶格 _1nn_list。
// 计算事件类型为 event::VacancyTrans，并设置相关的事件信息，包括起始晶格、目标晶格、目标标签。
// 使用 goto 标签 EVENT_FOUND 跳出循环。
// 如果以上两种类型都不是，表示事件未找到。
// 最后，通过 EVENT_FOUND 跳转标签，返回所选的事件。
// 该函数的目的是在指定的区域内选择一个事件，该事件的速率满足 excepted_rate，并返回一个包含事件信息的 event::SelectedEvent 对象。
// 这个函数通过遍历晶格、累加速率、检查类型等方式来选择事件，并在找到事件后使用 goto 跳转标签返回所选的事件。
// 如果没有找到满足条件的事件，函数将返回一个默认的 event::DefectGen 事件。

event::SelectedEvent ABVIModel::select(const lat_region region, const _type_rate excepted_rate,
                                       const _type_rate sum_rates) {
  _type_rate rate_accumulator = 0.0;
  event::SelectedEvent selected_event{event::DefectGen, 0, 0}; // default event is defect generation.
  for (const auto& it : box->lattice_list->vac_hash) {
  _type_lattice_size x = it.first % box->lattice_list->meta.size_x;
  _type_lattice_size y = (it.first / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
  _type_lattice_size z = it.first / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);
  if(2 * region.x_low <= x && x < 2 * region.x_high && region.y_low <= y && y < region.y_high && region.z_low <= z && z < region.z_high) {
// for (_type_lattice_size z = region.z_low; z < region.z_high; z++) {
//   for (_type_lattice_size y = region.y_low; y < region.y_high; y++) {
//     for (_type_lattice_size x = 2 * region.x_low; x < 2 * region.x_high; x++) {

    _type_lattice_size latti_id = box->lattice_list->getId(x, y, z);
    //if (box->lattice_list->vac_hash.count(latti_id)) { // vacancy
    const VacancyHash &vacancy = box->lattice_list->vac_hash.at(latti_id);
    for (int rate_index = 0; rate_index < Vacancy::RATES_SIZE; rate_index++) {
      rate_accumulator += vacancy.nn_rates[rate_index];
      //kiwi::logs::v(" ", " {} rates is : {} .\n", rate_index, vacancy.nn_rates[rate_index]);
      if (rate_accumulator > excepted_rate) {
        // Lattice *_1nn_list[LatticesList::MAX_1NN] = {nullptr};
        // box->lattice_list->get1nn(lattice.getId(), _1nn_list);
        selected_event.event_type = event::VacancyTrans;
        selected_event.from_id = latti_id;
        selected_event.to_id = box->lattice_list->meta.getIdBy1nnOffset(latti_id, rate_index);
        // assert(selected_event.to_id == _1nn_list[rate_index]->getId());
        selected_event.target_tag = static_cast<_type_dir_id>(rate_index);
        goto EVENT_FOUND; // event found
      }
      // event not found
    }
  }
}
EVENT_FOUND:
// kiwi::logs::v(" ", " excepted_rate is : {} . \n", excepted_rate);
#ifdef KMC_DEBUG_MODE
  //    todo assert total rates == rate_accumulator + defect_gen_rate
#endif
  return selected_event;
}
// 使用 switch 语句根据所选事件的类型进行不同的操作。
// 对于不同类型的事件，分别执行以下操作：
// event::VacancyTrans 事件：
// 获取起始晶格 lat_from 和目标晶格 lat_to。
// 交换晶格类型，即将 lat_from 的类型与 lat_to 的类型互换。
// 更新空位列表 box->va_list，将 lat_from 移除并添加 lat_to。
// 如果存在事件监听器 p_event_listener，则调用 onVacancyTrans 回调通知事件发生。
// 进行重新组合（recombination）操作，创建 rec::RecList 对象，选择最小的重新组合事件并执行。
// event::DumbbellTrans 事件：
// 获取起始晶格 lat_from 和目标晶格 lat_to。
// 复制起始晶格的相关信息，并保存在 itl_copy 中。
// 通过计算得到跳跃原子 jump_atom，并更新晶格类型以实现原子的交换。
// 更新晶格的方向信息。
// 如果存在事件监听器 p_event_listener，则调用 onDumbbellTrans 回调通知事件发生。
// 进行重新组合（recombination）操作，创建 rec::RecList 对象，选择最小的重新组合事件并执行。
// event::DefectGen 事件：
// 通过随机选择一个起始晶格，搜索满足条件的晶格对 lat_1 和 lat_2。
// 随机选择一个晶格作为空位（V），另一个晶格作为 dumbbell，然后交换晶格类型。
// 如果存在事件监听器 p_event_listener，则调用 onDefectGenerate 回调通知事件发生。
// 更新晶格的方向信息和相应的列表。
// 该函数的目的是执行所选的事件，根据事件类型执行不同的操作，包括交换晶格类型、更新空位列表、通知事件监听器以及进行重新组合操作等。
// 这个函数的实现根据不同事件类型的特点，进行了相应的处理。

// event::VacancyTrans 事件：
// 这个事件表示一个空位（Vacancy）向一个晶格（Atom）的转移。
// 首先，获取起始晶格 lat_from 和目标晶格 lat_to。
// 接下来，交换这两个晶格的类型，即将 lat_from 的类型和 lat_to 的类型互换。
// 更新空位列表 box->va_list，将 lat_from 移除，并将 lat_to 添加到空位列表中，表示空位移动到了新的位置。
// 如果存在事件监听器 p_event_listener，则调用 onVacancyTrans 回调通知事件发生，传递有关事件的相关信息。
// 最后，执行重新组合（recombination）操作，创建 rec::RecList 对象，选择最小的重新组合事件并执行。这个步骤可能导致一些额外的晶格类型的变化。
// event::DumbbellTrans 事件：
// 这个事件表示一个 dumbbell（双原子晶格）向一个晶格（Atom）的转移。
// 类似于上述步骤，首先获取起始晶格 lat_from 和目标晶格 lat_to。
// 然后，复制起始晶格的相关信息，并保存在 itl_copy 中，以便稍后使用。
// 通过计算得到跳跃原子 jump_atom，并更新晶格类型以实现原子的交换。例如，可以将 lat_from 中的 jump_atom 移动到 lat_to 中，同时将 lat_from 中的其他原子移动到 lat_to 中。
// 更新晶格的方向信息，以反映跳跃动作。
// 如果存在事件监听器 p_event_listener，则调用 onDumbbellTrans 回调通知事件发生，传递与事件相关的信息，如起始晶格类型、目标晶格类型等。
// 最后，执行重新组合（recombination）操作，创建 rec::RecList 对象，选择最小的重新组合事件并执行。这个步骤可能导致一些额外的晶格类型的变化。
// event::DefectGen 事件：
// 这个事件表示在晶格中生成一个缺陷。
// 通过随机选择一个起始晶格 lat_1，然后搜索附近的晶格以找到满足条件的 lat_2。
// 随机选择一个晶格作为空位（V），另一个晶格作为 dumbbell，然后交换它们的类型，以模拟缺陷的生成。
// 如果存在事件监听器 p_event_listener，则调用 onDefectGenerate 回调通知事件发生，传递与事件相关的信息，如起始晶格类型、目标晶格类型等。
// 更新晶格的方向信息，通常使用随机生成的方向。
// 最后，将生成的晶格和缺陷添加到相应的列表中，以反映新的晶格状态。
// 总之，perform 函数的目的是在不同的事件类型下执行相应的操作，以模拟晶格中事件的发生和影响。
// 这些事件可能涉及晶格类型的交换、空位的生成或移动、重新组合等，这些操作是基于模型和物理规律来执行的，以模拟晶格的动态行为。

void ABVIModel::perform(const event::SelectedEvent selected, const lat_region region, int rank, _type_lattice_count step, int sect, const unsigned int sector_id, const unsigned int next_sector_id) {
  // 获取转移前lattice的信息 引用！！
  // kiwi::logs::v(" ", " step is : {} sect is : {} from_id is : {} to_id is : {}. \n", step, sect, selected.from_id, selected.to_id);
  if(selected.event_type == event::VacancyTrans) {

    box->lattice_list->vac_hash.erase(selected.from_id);
    box->lattice_list->vac_hash.emplace(std::make_pair(selected.to_id, VacancyHash{}));

    if (box->lattice_list->re_hash.count(selected.to_id)) {
      box->lattice_list->re_hash.erase(selected.to_id);
      box->lattice_list->re_hash.emplace(selected.from_id);
    } else if (box->lattice_list->mn_hash.count(selected.to_id)) {
      box->lattice_list->mn_hash.erase(selected.to_id);
      box->lattice_list->mn_hash.emplace(selected.from_id);
    } else if (box->lattice_list->ni_hash.count(selected.to_id)) {
      box->lattice_list->ni_hash.erase(selected.to_id);
      box->lattice_list->ni_hash.emplace(selected.from_id);
    } else if (box->lattice_list->si_hash.count(selected.to_id)) {
      box->lattice_list->si_hash.erase(selected.to_id);
      box->lattice_list->si_hash.emplace(selected.from_id);
    }
    if (box->lattice_list->meta.isGhostLat(selected.to_id)) {
      // Lattice lat_to;
      // lat_to.id = selected.to_id;
      // lat_to.type = LatticeTypes{LatticeTypes::V};
      // lat_to.type = box->lattice_list->getType(selected.to_id);
      addExchange_ghost(selected.to_id, sector_id);
    }
    if (box->lattice_list->meta.isSurfaceLat(selected.from_id)) addExchange_surface(selected.from_id);
    if (box->lattice_list->meta.isSurfaceLat(selected.to_id)) addExchange_surface(selected.to_id);
  } else assert(false);
}

void ABVIModel::recb_checki(const lat_region region, const unsigned int sector_id) {
  // kiwi::logs::v(" ", " Before {} , momo_count : {} , more_count : {}, rere_count : {}.\n", sector_id, box->lattice_list->momo_hash.size(), box->lattice_list->more_hash.size(), box->lattice_list->rere_hash.size());
  // kiwi::logs::v(" ", "start MoRe GPU Version");
  std::vector<int> arr;
  int arr_index = 0;

  for (const auto& it : box->lattice_list->momo_hash) {
    _type_lattice_size x = it % box->lattice_list->meta.size_x;
    _type_lattice_size y = (it / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
    _type_lattice_size z = it / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);
    if(2 * region.x_low <= x && x < 2 * region.x_high && region.y_low <= y && y < region.y_high && region.z_low <= z && z < region.z_high) {
      arr.emplace_back(it);
    }
  }

  for (const auto& it : box->lattice_list->more_hash) {
    _type_lattice_size x = it % box->lattice_list->meta.size_x;
    _type_lattice_size y = (it / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
    _type_lattice_size z = it / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);
    if(2 * region.x_low <= x && x < 2 * region.x_high && region.y_low <= y && y < region.y_high && region.z_low <= z && z < region.z_high) {
      arr.emplace_back(it);
    }
  }

  for (const auto& it : box->lattice_list->rere_hash) {
    _type_lattice_size x = it % box->lattice_list->meta.size_x;
    _type_lattice_size y = (it / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
    _type_lattice_size z = it / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);
    if(2 * region.x_low <= x && x < 2 * region.x_high && region.y_low <= y && y < region.y_high && region.z_low <= z && z < region.z_high) {
      arr.emplace_back(it);
    }
  }
  /*
    本部分用于图着色
   */
  int n = arr.size();
  // std::vector<int> colors(n , -1);//colors 用于保存目前的着色情况
  std::unordered_map<_type_lattice_id, int> color_hash;
  for(int i=0;i<n;i++){
    color_hash[arr[i]] = -1;
  }

  for(int i = 0;i < n; i++){
      // if(colors[i] == -1)
      //   colors[i] = 0;//如果未着色则着色为0
      _type_lattice_size x = arr[i] % box->lattice_list->meta.size_x;
      _type_lattice_size y = (arr[i] / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
      _type_lattice_size z = arr[i] / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);

      Lattice nn_list1[LatticesList::MAX_1NN];//当前遍历的list的1nn近邻
      Lattice nn_list2[LatticesList::MAX_2NN];//当前遍历的list的2nn近邻
      Lattice nn_list[LatticesList::MAX_1NN + LatticesList::MAX_2NN ];


      std::vector<bool> avail_colors(LatticesList::MAX_1NN + LatticesList::MAX_2NN,true);//当前着色区域内的可用颜色
      // avail_colors[color_hash[arr[i]]] = false; //将目前颜色设置为否


      box->lattice_list->get1nn2(x, y, z, nn_list1);//TODO:不知道这个函数是用于1nn近邻还是2nn
      box->lattice_list->get2nn2(x, y, z, nn_list2);
      //将两个数组拼接
      for (int i=0;i<LatticesList::MAX_1NN;i++)
        nn_list[i] = nn_list1[i];
      for (int i=0;i<LatticesList::MAX_2NN;i++)
        nn_list[i + LatticesList::MAX_1NN] = nn_list2[i];


      // for(int k =0;k<LatticesList::MAX_1NN + LatticesList::MAX_2NN; k++){
      //     kiwi::logs::v(" ","1color {} -- {}\n",k,avail_colors[k]);
      // }
      if(color_hash[arr[i]] != -1){
        avail_colors[color_hash[arr[i]]] = false;
      }
      // for(int k =0;k<LatticesList::MAX_1NN + LatticesList::MAX_2NN; k++){
      //     kiwi::logs::v(" ","2color {} -- {}\n",k,avail_colors[k]);
      // }
        for (const auto& lati : nn_list) {//遍历1nn和2nn列表，看arr[j]是否在arr[i]的2nn近邻以内
          if(color_hash.find(lati.getId()) != color_hash.end()){//证明此位置是一个元素
            // kiwi::logs::v(" ","atom {}----- atom{}\n",arr[i],lati.getId());
            if(color_hash[lati.getId()] != -1){//已经被上过色
              avail_colors[color_hash[lati.getId()]] = false;//则此颜色不可用
            }
          }
        }
      // for(int k =0;k<LatticesList::MAX_1NN + LatticesList::MAX_2NN; k++){
      //     kiwi::logs::v(" ","3color {} -- {}\n",k,avail_colors[k]);
      // }
      if(color_hash[arr[i]] == -1){
        for(int k =0; k < LatticesList::MAX_1NN + LatticesList::MAX_2NN; k++){//遍历可用色列表
            if(avail_colors[k] == true){
              color_hash[arr[i]] = k;
              avail_colors[k] = false;
              break;
            }
        }
      }
      // for(int k =0;k<LatticesList::MAX_1NN + LatticesList::MAX_2NN; k++){
      //     kiwi::logs::v(" ","4color {} -- {}\n",k,avail_colors[k]);
      // }
      for (const auto& lati : nn_list){
        if(color_hash.find(lati.getId()) != color_hash.end()){//证明此位置是一个元素
          if(color_hash[lati.getId()] == -1){
              for(int k =0; k < LatticesList::MAX_1NN + LatticesList::MAX_2NN; k++){//遍历可用色列表
                if(avail_colors[k] == true){
                  color_hash[lati.getId()] = k;
                  avail_colors[k] = false;
                  break;
              }
            }
          }
        }
      }
      // for(int k =0;k<LatticesList::MAX_1NN + LatticesList::MAX_2NN; k++){
      //     kiwi::logs::v(" ","5color {} -- {}\n",k,avail_colors[k]);
      // }
        
  }

  for(int i = 0;i<LatticesList::MAX_1NN + LatticesList::MAX_2NN;i++){//遍历各个颜色的数组
    std::vector<_type_lattice_id> color_arr;
    std::vector<int> color_arr_id;//用于存放当前颜色的id的列表
    std::vector<Lattice> nn_lists1;//用于存放每个原子的1nn和2nn近邻的信息
    std::vector<Lattice> nn_lists2;//用于存放每个原子的1nn和2nn近邻的信息
    for(int j = 0; j < arr.size();j++){
      if(color_hash[arr[j]] == i){
      _type_lattice_size x = arr[j] % box->lattice_list->meta.size_x;
      _type_lattice_size y = (arr[j] / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
      _type_lattice_size z = arr[j] / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);

      Lattice nn_list1[LatticesList::MAX_1NN];//当前遍历的list的1nn近邻
      Lattice nn_list2[LatticesList::MAX_2NN];//当前遍历的list的2nn近邻

      box->lattice_list->get1nn2(x, y, z, nn_list1);//TODO:不知道这个函数是用于1nn近邻还是2nn
      box->lattice_list->get2nn2(x, y, z, nn_list2);


    //   kiwi::logs::v(" ","---------------------------------------------\n");
      for(auto _1nn_atom:nn_list1){
        nn_lists1.push_back(_1nn_atom);
        // kiwi::logs::v(" ","{}-{}\n",_1nn_atom.id,_1nn_atom.type._type);
      }
        
      for(auto _2nn_atom:nn_list2){
        nn_lists1.push_back(_2nn_atom);
        // kiwi::logs::v(" ","{}-{}\n",_2nn_atom.id,_2nn_atom.type._type);
      }

    //   nn_lists1 = nn_lists1 + nn_lists2;
    //   kiwi::logs::v(" ","---------------------------------------------\n");
        

    // for(_type_lattice_count j =0; j < LatticesList::MAX_1NN; j++){
    //   kiwi::logs::v(" ","{}-{}\n",atoms[i].atom1nn[j],atoms[i].atom1nn_type[j]);
    // }
    // for(_type_lattice_count j =0; j < LatticesList::MAX_2NN; j++){
    //   kiwi::logs::v(" ","{}-{}\n",atoms[i].atom2nn[j],atoms[i].atom2nn_type[j]);
    // }

      

        color_arr.push_back(arr[j]);//将第i种颜色放入数组中
                                    //arr中存放的是每个color的hash值
        // color_arr_id.push_back(arr[j].first);    
      }
          
    }
    // kiwi::logs::v(" ","current color is {}:",i);
    // for(auto atom:color_arr)
    //   kiwi::logs::v(" ","{},",atom);
    // kiwi::logs::v(" ","\n");
    std::vector<_type_lattice_size>x_array;
    std::vector<_type_lattice_size>y_array;
    std::vector<_type_lattice_size>z_array;
    std::vector< LatticeTypes::lat_type> centerTypes;
    for(auto id:color_arr){
        _type_lattice_size x = id % box->lattice_list->meta.size_x;
        x_array.push_back(x);
        _type_lattice_size y = (id / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
        y_array.push_back(y);
        _type_lattice_size z = id / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);
        z_array.push_back(z);
        LatticeTypes::lat_type cur_type = box->lattice_list->getType(id);
        centerTypes.push_back(cur_type);
        // kiwi::logs::v(" ","cur_type:{}\n",cur_type);


    }
    // kiwi::logs::v(" ","current color is {}: size is {}\n",i,color_arr.size());
    if(color_arr.size() > 0){
      dev_atom *atoms = recb_solver_GPU(color_arr,centerTypes,color_arr.size(),nn_lists1,nn_lists2,sector_id,x_array,y_array,z_array);//TODO:设计输入参数和GPU内容
      std::vector<int> res;
      for(int i =0;i<color_arr.size();i++){
         auto it = std::find(res.begin(), res.end(), atoms[i].exchange_id);
         if(it!= res.end())
          kiwi::logs::v(" ","conflict occured {}\n",atoms[i].exchange_id);
         else
          res.push_back(atoms[i].exchange_id);
      }
        
      
      // for(int k = 0;k<color_arr.size();k++){
      //   if(atoms[k].exchange == 1)
      //     kiwi::logs::v(" ","atom type is {},exchange type:{}\n",atoms[k].atom_id,atoms[k].exchange,atoms[k].exchange_id);
      // }
      for (int i = 0;i<color_arr.size();i++){
        // kiwi::logs::v(" ","id------:{} - {}\n",atoms[i].atom_id, lat::LatTypesString(box->lattice_list->getType(atoms[i].atom_id)));
        // kiwi::logs::v(" ","exchange:{} - {}\n",atoms[i].exchange_id, lat::LatTypesString(box->lattice_list->getType(atoms[i].exchange_id)));

          if(atoms[i].exchange == 1 && atoms[i].out_sector == 0){
            // kiwi::logs::v(" ","exchange occured\n");
            _type_lattice_size x = atoms[i].exchange_id % box->lattice_list->meta.size_x;
            _type_lattice_size y = (atoms[i].exchange_id  / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
            _type_lattice_size z = atoms[i].exchange_id  / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);
            if(atoms[i].exchange_type == LatticeTypes::Mo || atoms[i].exchange_type == LatticeTypes::Re){
              if(atoms[i].atom_type == LatticeTypes::MoMo){
                if(atoms[i].exchange_type ==  LatticeTypes::Mo){
                  box->lattice_list->momo_hash.erase(atoms[i].atom_id);
                  box->lattice_list->momo_hash.emplace(atoms[i].exchange_id);
                  // atoms[i].exchange_type = box->lattice_list->getType(atoms[i].exchange_id);
                  // atoms[i].atom_type = LatticeTypes::Mo;
                  // kiwi::logs::v(" ","{}-{}-{} ::change_id changed {}->{}-{}\n",atoms[i].atom_id,lat::LatTypesString(atoms[i].atom_type),lat::LatTypesString(box->lattice_list->getType(atoms[i].atom_id))
                  //               ,atoms[i].exchange_id,lat::LatTypesString(atoms[i].exchange_type),lat::LatTypesString(box->lattice_list->getType(atoms[i].exchange_id)));
                  
                } 
                else if(atoms[i].exchange_type == LatticeTypes::Re){
                  box->lattice_list->re_hash.erase(atoms[i].exchange_id);
                  box->lattice_list->momo_hash.erase(atoms[i].atom_id);
                  auto res = box->lattice_list->more_hash.emplace(atoms[i].exchange_id);
                  // kiwi::logs::v(" ","{}-{}-{} ::change_id changed {}->{}-{}\n",atoms[i].atom_id,lat::LatTypesString(atoms[i].atom_type),lat::LatTypesString(box->lattice_list->getType(atoms[i].atom_id))
                  //               ,atoms[i].exchange_id,lat::LatTypesString(atoms[i].exchange_type),lat::LatTypesString(box->lattice_list->getType(atoms[i].exchange_id)));

                  // atoms[i].exchange_type = box->lattice_list->getType(atoms[i].exchange_id);
                  // atoms[i].atom_type = LatticeTypes::Mo;

                  if (!res.second) {
                        kiwi::logs::v(" ","key exists\n");
                    }
                //   atoms[i].atom_type =  LatticeTypes::MoRe;
                }
                if (box->lattice_list->meta.isGhostLat(atoms[i].exchange_id)) {
                  addExchange_ghost(atoms[i].exchange_id, sector_id);
                }
                if (box->lattice_list->meta.isSurfaceLat(atoms[i].exchange_id)) 
                  addExchange_surface(atoms[i].exchange_id);
                if (box->lattice_list->meta.isSurfaceLat(atoms[i].atom_id)) 
                  addExchange_surface(atoms[i].atom_id);
              }
              else if(atoms[i].atom_type = LatticeTypes::MoRe){
                if(atoms[i].numRe == 0){
                  if (atoms[i].exchange_type == LatticeTypes::Mo) { // 交换位置
                    // kiwi::logs::v(" ","key  not exixt\n");
                    // if(box->lattice_list->more_hash.count(atoms[i].atom_id)== 0)
                    //     kiwi::logs::v(" ","{} :key  not exist\n",atoms[i].atom_id);
                    box->lattice_list->more_hash.erase(atoms[i].atom_id);
                    auto res = box->lattice_list->more_hash.emplace(atoms[i].exchange_id);
                    // kiwi::logs::v(" ","{}-{}-{} ::change_id changed {}->{}-{}\n",atoms[i].atom_id,lat::LatTypesString(atoms[i].atom_type),lat::LatTypesString(box->lattice_list->getType(atoms[i].atom_id))
                    //             ,atoms[i].exchange_id,lat::LatTypesString(atoms[i].exchange_type),lat::LatTypesString(box->lattice_list->getType(atoms[i].exchange_id)));
                    // atoms[i].exchange_type = box->lattice_list->getType(atoms[i].exchange_id);
                    // atoms[i].atom_type = LatticeTypes::Mo;
                      if (!res.second) {
                        kiwi::logs::v(" ","{}::{} :key exists - {} - {}\n",atoms[i].atom_id,atoms[i].exchange_id, lat::LatTypesString(box->lattice_list->getType(atoms[i].exchange_id)),lat::LatTypesString(box->lattice_list->getType(atoms[i].exchange_type )));
                    }
                  } else { // Re 也是交换位置
                    box->lattice_list->re_hash.erase(atoms[i].exchange_id);
                    box->lattice_list->more_hash.erase(atoms[i].atom_id);
                    box->lattice_list->re_hash.emplace(atoms[i].atom_id);
                    box->lattice_list->more_hash.emplace(atoms[i].exchange_id);
                    // kiwi::logs::v(" ","{}-{}-{} ::change_id changed {}->{}-{}\n",atoms[i].atom_id,lat::LatTypesString(atoms[i].atom_type),lat::LatTypesString(box->lattice_list->getType(atoms[i].atom_id))
                    //             ,atoms[i].exchange_id,lat::LatTypesString(atoms[i].exchange_type),lat::LatTypesString(box->lattice_list->getType(atoms[i].exchange_id)));
                    // atoms[i].exchange_type = box->lattice_list->getType(atoms[i].exchange_id);
                    // atoms[i].atom_type = LatticeTypes::Re;
                  }
                  if (box->lattice_list->meta.isGhostLat(atoms[i].exchange_id)) {
                    addExchange_ghost(atoms[i].exchange_id,sector_id);
                  }
                  if (box->lattice_list->meta.isSurfaceLat(atoms[i].exchange_id)) 
                    addExchange_surface(atoms[i].exchange_id);
                  if (box->lattice_list->meta.isSurfaceLat(atoms[i].atom_id)) 
                    addExchange_surface(atoms[i].atom_id);
                }
                else if(atoms[i].numRe > 0 && atoms[i].numRe < LatticesList::MAX_1NN ){
                  if(atoms[i].ranmov > (static_cast<double>(atoms[i].numRe) / LatticesList::MAX_1NN)){
                  if (atoms[i].exchange_type == LatticeTypes::Mo) {  // 交换位置
                    box->lattice_list->more_hash.erase(atoms[i].atom_id);
                    box->lattice_list->more_hash.emplace(atoms[i].exchange_id);

                    atoms[i].exchange_type = box->lattice_list->getType(atoms[i].exchange_id);
                    atoms[i].atom_type = LatticeTypes::Mo;
                    // kiwi::logs::v(" ","{}-{}-{} ::change_id changed {}->{}-{}\n",atoms[i].atom_id,lat::LatTypesString(atoms[i].atom_type),lat::LatTypesString(box->lattice_list->getType(atoms[i].atom_id))
                    //             ,atoms[i].exchange_id,lat::LatTypesString(atoms[i].exchange_type),lat::LatTypesString(box->lattice_list->getType(atoms[i].exchange_id)));
                  } else if(atoms[i].exchange_type == LatticeTypes::Re) { // Re 也是交换位置
                    box->lattice_list->re_hash.erase(atoms[i].exchange_id);
                    box->lattice_list->more_hash.erase(atoms[i].atom_id);
                    box->lattice_list->re_hash.emplace(atoms[i].atom_id);
                    box->lattice_list->more_hash.emplace(atoms[i].exchange_id);

                    // atoms[i].exchange_type = box->lattice_list->getType(atoms[i].exchange_id);
                    // atoms[i].atom_type = LatticeTypes::Re;
                    // kiwi::logs::v(" ","{}-{}-{} ::change_id changed {}->{}-{}\n",atoms[i].atom_id,lat::LatTypesString(atoms[i].atom_type),lat::LatTypesString(box->lattice_list->getType(atoms[i].atom_id))
                    //             ,atoms[i].exchange_id,lat::LatTypesString(atoms[i].exchange_type),lat::LatTypesString(box->lattice_list->getType(atoms[i].exchange_id)));
                  }
                  if (box->lattice_list->meta.isGhostLat(atoms[i].exchange_id)) {
                    addExchange_ghost(atoms[i].exchange_id, sector_id);
                  }
                  if (box->lattice_list->meta.isSurfaceLat(atoms[i].exchange_id)) addExchange_surface(atoms[i].exchange_id);
                  if (box->lattice_list->meta.isSurfaceLat(atoms[i].atom_id)) addExchange_surface(atoms[i].atom_id);
              }
            }
            // else return;
          }
        }

        if(2 * region.x_low <= x && x < 2 * region.x_high && region.y_low <= y && y < region.y_high && region.z_low <= z && z < region.z_high){
            atoms[i].out_sector = 1;
            // kiwi::logs::v(" ","id{} - out sector\n",atoms[i].exchange_id);
        }
        
        atoms[i].exchange_type = box->lattice_list->getType(atoms[i].exchange_id);
        atoms[i].atom_type =  box->lattice_list->getType(atoms[i].atom_id);
      
      }
        //   else continue;
      }
    }
  }

  //图着色结束以后根据着色的结果进行并行化，各组颜色执行完以后为一组，然后重新着色，进行下一次迭代
  //但是仍然需要注意移出区域后的判断
  // for (const auto& it : arr) {
  //   recb_solver(it, region, sector_id);
  // }
  //这是之前的非GPU版本的

  // kiwi::logs::v(" ", " After {} , momo_count : {} , more_count : {}, rere_count : {}.\n", sector_id, box->lattice_list->momo_hash.size(), box->lattice_list->more_hash.size(), box->lattice_list->rere_hash.size());
}

void ABVIModel::recb_solver(_type_lattice_id id, const lat_region& region, const unsigned int& sector_id) {
  
  LatticeTypes::lat_type cur_type = box->lattice_list->getType(id); 
  _type_lattice_size x = id % box->lattice_list->meta.size_x;
  _type_lattice_size y = (id / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
  _type_lattice_size z = id / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);

  // bool trapped = false;
  // bool trapped_by_solute = false;
  int numSIA;
  int numRe;
  Lattice nn_list[LatticesList::MAX_1NN];
  _type_lattice_size temp_x, temp_y, temp_z;
  Lattice randomLattice;

  //这个while会产生团簇
  while (2 * region.x_low <= x && x < 2 * region.x_high && region.y_low <= y && y < region.y_high && region.z_low <= z && z < region.z_high) {
    id = box->lattice_list->getId(x, y, z);
    assert(cur_type == box->lattice_list->getType(x, y, z));

    numSIA = 0;
    box->lattice_list->get1nn2(x, y, z, nn_list);
    for (const auto& lati : nn_list) {
      if (lati.type._type == LatticeTypes::MoMo || lati.type._type == LatticeTypes::MoRe) numSIA++;
    }
    if (numSIA >= 2) return;

    numRe = 0;
    if (cur_type == LatticeTypes::MoRe) {
      for (const auto& lati : nn_list) {
        if (lati.type._type == LatticeTypes::Re) numRe++;
      }
      if (numRe >= 5) return;
    }

    // 2nn 内随机运动
    int randomNumber = static_cast<int>(rand() * 14);
    assert(randomNumber >= 0 && randomNumber < 14);
    box->lattice_list->getRandomLattice(x, y, z, temp_x, temp_y, temp_z, randomNumber);

    // 当前扇区内随机运动
    // temp_x = static_cast<_type_lattice_size>(rand() * 2 * (region.x_high - regin.x_low) + 2 * region.x_low);
    // temp_y = static_cast<_type_lattice_size>(rand() * (region.y_high - region.y_low) + region.y_low);
    // temp_z = static_cast<_type_lattice_size>(rand() * (region.z_high - region.z_low) + region.z_low);

    randomLattice.id = box->lattice_list->getId(temp_x, temp_y, temp_z);
    randomLattice.type._type = box->lattice_list->getType(randomLattice.id);
    
    // kiwi::logs::v(" ", " temp_x is : {} temp_y is : {} temp_z is : {} type is : {}.\n", temp_x, temp_y, temp_z, randomLattice.type._type);
    if (randomLattice.type._type == LatticeTypes::Mo || randomLattice.type._type == LatticeTypes::Re) {
      if (cur_type == LatticeTypes::MoMo) {
        if (randomLattice.type._type == LatticeTypes::Mo) { // 交换位置
          box->lattice_list->momo_hash.erase(id);
          box->lattice_list->momo_hash.emplace(randomLattice.id);
        } else if(randomLattice.type._type == LatticeTypes::Re) { // Re (momo 变成 mo，re 变成 more)
          box->lattice_list->re_hash.erase(randomLattice.id);
          box->lattice_list->momo_hash.erase(id);
          box->lattice_list->more_hash.emplace(randomLattice.id);
          cur_type = LatticeTypes::MoRe;
        }
        x = temp_x;
        y = temp_y;
        z = temp_z;
        if (box->lattice_list->meta.isGhostLat(randomLattice.id)) {
          addExchange_ghost(randomLattice.id, sector_id);
        }
        if (box->lattice_list->meta.isSurfaceLat(randomLattice.id))
          addExchange_surface(randomLattice.id);
        if (box->lattice_list->meta.isSurfaceLat(id)) 
          addExchange_surface(id);
      } else if (cur_type == LatticeTypes::MoRe) {
        if (numRe == 0) {
          if (randomLattice.type._type == LatticeTypes::Mo) { // 交换位置
            box->lattice_list->more_hash.erase(id);
            box->lattice_list->more_hash.emplace(randomLattice.id);
          } else { // Re 也是交换位置
            box->lattice_list->re_hash.erase(randomLattice.id);
            box->lattice_list->more_hash.erase(id);
            box->lattice_list->re_hash.emplace(id);
            box->lattice_list->more_hash.emplace(randomLattice.id);
          }
          x = temp_x;
          y = temp_y;
          z = temp_z;

          if (box->lattice_list->meta.isGhostLat(randomLattice.id)) {
            addExchange_ghost(randomLattice.id, sector_id);
          }
          if (box->lattice_list->meta.isSurfaceLat(randomLattice.id)) 
            addExchange_surface(randomLattice.id);
          if (box->lattice_list->meta.isSurfaceLat(id)) addExchange_surface(id);
        } else if (numRe > 0 && numRe < LatticesList::MAX_1NN) {
          double ranmov = rand();
          if(ranmov > (static_cast<double>(numRe) / LatticesList::MAX_1NN)) {
            if (randomLattice.type._type == LatticeTypes::Mo) {  // 交换位置
              box->lattice_list->more_hash.erase(id);
              box->lattice_list->more_hash.emplace(randomLattice.id);
            } else if(randomLattice.type._type == LatticeTypes::Re) { // Re 也是交换位置
              box->lattice_list->re_hash.erase(randomLattice.id);
              box->lattice_list->more_hash.erase(id);
              box->lattice_list->re_hash.emplace(id);
              // box->lattice_list->more_hash.emplace(randomLattice.id);
            }
            x = temp_x;
            y = temp_y;
            z = temp_z;

            if (box->lattice_list->meta.isGhostLat(randomLattice.id)) {
              addExchange_ghost(randomLattice.id, sector_id);
            }
            if (box->lattice_list->meta.isSurfaceLat(randomLattice.id)) addExchange_surface(randomLattice.id);
            if (box->lattice_list->meta.isSurfaceLat(id)) addExchange_surface(id);
          }
        } else return; // if (numRed >= LatticesList::MAX_1NN)
      }
    } else continue;

  }
}

void ABVIModel::reindex(const lat_region region) {
  box->va_list->reindex(box->lattice_list, region);
  // todo reindex interval list
  // todo refresh counter
}

void ABVIModel::setEventListener(EventListener *p_listener) { p_event_listener = p_listener; }

void ABVIModel::setColoredDomain(comm::ColoredDomain *_p_domain) { p_domain = _p_domain; }

void ABVIModel::set_ghost_region() {
  for(int i = 0; i < 8; i++) {
    switch (i)
    {
      case 0:
        // 不需要存储转发 直接向下
        ghost_region[i * 7 + 0].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 0].y_low = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 0].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 0].x_high = 2 * (p_domain->local_split_coord[0] + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 0].y_high = p_domain->local_split_coord[1] + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 0].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 需要存储转发1次 先向下，再向前
        ghost_region[i * 7 + 1].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 1].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 1].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 1].x_high = 2 * (p_domain->local_split_coord[0] + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 1].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 1].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 需要存储转发1次 先向下，再向左
        ghost_region[i * 7 + 2].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 2].y_low = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 2].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 2].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 2].y_high = p_domain->local_split_coord[1] + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 2].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 需要存储转发2次 先向下，再向前，最后向左
        ghost_region[i * 7 + 3].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 3].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 3].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 3].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 3].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 3].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 不需要存储转发 直接向前
        ghost_region[i * 7 + 4].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 4].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 4].z_low = p_domain->local_sub_box_lattice_region.z_low; // 2
        ghost_region[i * 7 + 4].x_high = 2 * (p_domain->local_split_coord[0] + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 4].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 4].z_high = p_domain->local_split_coord[2] + p_domain->lattice_size_ghost[2]; 

        // 需要存储转发1次 先向前，再向左
        ghost_region[i * 7 + 5].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 5].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 5].z_low = p_domain->local_sub_box_lattice_region.z_low; // 2
        ghost_region[i * 7 + 5].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 5].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 5].z_high = p_domain->local_split_coord[2] + p_domain->lattice_size_ghost[2];

        // 不需要存储转发 直接向左
        ghost_region[i * 7 + 6].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 6].y_low = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 6].z_low = p_domain->local_sub_box_lattice_region.z_low; // 2
        ghost_region[i * 7 + 6].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 6].y_high = p_domain->local_split_coord[1] + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 6].z_high = p_domain->local_split_coord[2] + p_domain->lattice_size_ghost[2]; 
        break;
      case 1:
        // 不需要存储转发 直接向下
        ghost_region[i * 7 + 0].x_low = 2 * (p_domain->local_split_coord[0] - p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 0].y_low = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 0].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 0].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 0].y_high = p_domain->local_split_coord[1] + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 0].z_high = p_domain->local_sub_box_lattice_region.x_low; // 2

        //  需要存储转发1次 先向下，再向前
        ghost_region[i * 7 + 1].x_low = 2 * (p_domain->local_split_coord[0] - p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 1].y_low = p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 1].z_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 1].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 1].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 1].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 需要存储转发1次 先向下，再向右
        ghost_region[i * 7 + 2].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 2].y_low = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 2].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 2].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 2].y_high = p_domain->local_split_coord[1] + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 2].z_high = p_domain->local_sub_box_lattice_region.x_low; // 2

        // 需要存储转发2次 先向下，再向前，最后向右
        ghost_region[i * 7 + 3].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 3].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 3].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 3].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 3].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 3].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 不需要存储转发 直接向前
        ghost_region[i * 7 + 4].x_low = 2 * (p_domain->local_split_coord[0] - p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 4].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 4].z_low = p_domain->local_sub_box_lattice_region.z_low; // 2
        ghost_region[i * 7 + 4].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 4].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 4].z_high = p_domain->local_split_coord[2] + p_domain->lattice_size_ghost[2]; 

        // 需要存储转发1次 先向前，再向右
        ghost_region[i * 7 + 5].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 5].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 5].z_low = p_domain->local_sub_box_lattice_region.z_low; // 2
        ghost_region[i * 7 + 5].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 5].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 5].z_high = p_domain->local_split_coord[2] + p_domain->lattice_size_ghost[2]; 

        // 不需要存储转发 直接向右
        ghost_region[i * 7 + 6].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high); 
        ghost_region[i * 7 + 6].y_low = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 6].z_low = p_domain->local_sub_box_lattice_region.z_low; // 2
        ghost_region[i * 7 + 6].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 6].y_high = p_domain->local_split_coord[1] + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 6].z_high = p_domain->local_split_coord[2] + p_domain->lattice_size_ghost[2]; 
        break;
      case 2:
        // 不需要存储转发 直接向下
        ghost_region[i * 7 + 0].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 0].y_low = p_domain->local_split_coord[1] - p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 0].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 0].x_high = 2 * (p_domain->local_split_coord[0] + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 0].y_high = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 0].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 需要存储转发1次 先向下，再向后
        ghost_region[i * 7 + 1].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 1].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 1].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 1].x_high = 2 * (p_domain->local_split_coord[0] + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 1].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 1].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 需要存储转发1次 先向下，再向左
        ghost_region[i * 7 + 2].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 2].y_low = p_domain->local_split_coord[1] - p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 2].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 2].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 2].y_high = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 2].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 需要存储转发2次 先向下，再向后，最后向左
        ghost_region[i * 7 + 3].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 3].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 3].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 3].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 3].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 3].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 不需要存储转发 直接向后
        ghost_region[i * 7 + 4].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 4].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 4].z_low = p_domain->local_sub_box_lattice_region.z_low; // 2
        ghost_region[i * 7 + 4].x_high = 2 * (p_domain->local_split_coord[0] + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 4].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 4].z_high = p_domain->local_split_coord[2] + p_domain->lattice_size_ghost[2]; 

        // 需要存储转发1次 先向后，再向左
        ghost_region[i * 7 + 5].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 5].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 5].z_low = p_domain->local_sub_box_lattice_region.z_low; // 2
        ghost_region[i * 7 + 5].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 5].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 5].z_high = p_domain->local_split_coord[2] + p_domain->lattice_size_ghost[2]; 

        // 不需要存储转发 直接向左
        ghost_region[i * 7 + 6].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 6].y_low = p_domain->local_split_coord[1] - p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 6].z_low = p_domain->local_sub_box_lattice_region.z_low; // 2
        ghost_region[i * 7 + 6].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 6].y_high = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 6].z_high = p_domain->local_split_coord[2] + p_domain->lattice_size_ghost[2]; 
        break;
      case 3:
        // 不需要存储转发 直接向下
        ghost_region[i * 7 + 0].x_low = 2 * (p_domain->local_split_coord[0] - p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 0].y_low = p_domain->local_split_coord[1] - p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 0].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 0].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 0].y_high = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 0].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 需要存储转发1次 先向下，再向后
        ghost_region[i * 7 + 1].x_low = 2 * (p_domain->local_split_coord[0] - p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 1].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 1].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 1].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 1].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 1].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 需要存储转发1次 先向下，再向右
        ghost_region[i * 7 + 2].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 2].y_low = p_domain->local_split_coord[1] - p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 2].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 2].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 2].y_high = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 2].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 需要存储转发2次 先向下，再向后，最后向右
        ghost_region[i * 7 + 3].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 3].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 3].z_low = p_domain->local_sub_box_lattice_region.z_low - p_domain->lattice_size_ghost[2]; // 0
        ghost_region[i * 7 + 3].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 3].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 3].z_high = p_domain->local_sub_box_lattice_region.z_low; // 2

        // 不需要存储转发 直接向后
        ghost_region[i * 7 + 4].x_low = 2 * (p_domain->local_split_coord[0] - p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 4].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 4].z_low = p_domain->local_sub_box_lattice_region.z_low; // 2
        ghost_region[i * 7 + 4].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 4].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 4].z_high = p_domain->local_split_coord[2] + p_domain->lattice_size_ghost[2]; 

        // 需要存储转发1次 先向后，再向右
        ghost_region[i * 7 + 5].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 5].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 5].z_low = p_domain->local_sub_box_lattice_region.z_low; // 2
        ghost_region[i * 7 + 5].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 5].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 5].z_high = p_domain->local_split_coord[2] + p_domain->lattice_size_ghost[2]; 

        // 不需要存储转发 直接向右
        ghost_region[i * 7 + 6].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high); 
        ghost_region[i * 7 + 6].y_low = p_domain->local_split_coord[1] - p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 6].z_low = p_domain->local_sub_box_lattice_region.z_low; // 2
        ghost_region[i * 7 + 6].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 6].y_high = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 6].z_high = p_domain->local_split_coord[2] + p_domain->lattice_size_ghost[2]; 
        break;
      case 4:
        // 不需要存储转发 直接向上
        ghost_region[i * 7 + 0].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 0].y_low = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 0].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 0].x_high = 2 * (p_domain->local_split_coord[0] + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 0].y_high = p_domain->local_split_coord[1] + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 0].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 需要存储转发1次 先向上，再向前
        ghost_region[i * 7 + 1].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 1].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 1].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 1].x_high = 2 * (p_domain->local_split_coord[0] + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 1].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 1].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];
 
        // 需要存储转发1次 先向上，再向左
        ghost_region[i * 7 + 2].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 2].y_low = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 2].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 2].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 2].y_high = p_domain->local_split_coord[1] + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 2].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 需要存储转发2次 先向上，再向前，最后向左
        ghost_region[i * 7 + 3].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 3].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 3].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 3].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 3].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 3].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 不需要存储转发 直接向前
        ghost_region[i * 7 + 4].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 4].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 4].z_low = p_domain->local_split_coord[2] - p_domain->lattice_size_ghost[2]; 
        ghost_region[i * 7 + 4].x_high = 2 * (p_domain->local_split_coord[0] + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 4].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 4].z_high = p_domain->local_sub_box_lattice_region.z_high;

        // 需要存储转发1次 先向前，再向左
        ghost_region[i * 7 + 5].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 5].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 5].z_low = p_domain->local_split_coord[2] - p_domain->lattice_size_ghost[2]; 
        ghost_region[i * 7 + 5].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 5].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 5].z_high = p_domain->local_sub_box_lattice_region.z_high;

        // 不需要存储转发 直接向左
        ghost_region[i * 7 + 6].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 6].y_low = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 6].z_low = p_domain->local_split_coord[2] - p_domain->lattice_size_ghost[2]; 
        ghost_region[i * 7 + 6].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 6].y_high = p_domain->local_split_coord[1] + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 6].z_high = p_domain->local_sub_box_lattice_region.z_high;
        break;
      case 5:
        // 不需要存储转发 直接向上
        ghost_region[i * 7 + 0].x_low = 2 * (p_domain->local_split_coord[0] - p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 0].y_low = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 0].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 0].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 0].y_high = p_domain->local_split_coord[1] + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 0].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        //  需要存储转发1次 先向上，再向前
        ghost_region[i * 7 + 1].x_low = 2 * (p_domain->local_split_coord[0] - p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 1].y_low = p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 1].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 1].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 1].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 1].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 需要存储转发1次 先向上，再向右
        ghost_region[i * 7 + 2].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 2].y_low = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 2].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 2].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 2].y_high = p_domain->local_split_coord[1] + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 2].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 需要存储转发2次 先向上，再向前，最后向右
        ghost_region[i * 7 + 3].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 3].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 3].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 3].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 3].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 3].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 不需要存储转发 直接向前
        ghost_region[i * 7 + 4].x_low = 2 * (p_domain->local_split_coord[0] - p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 4].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 4].z_low = p_domain->local_split_coord[2] - p_domain->lattice_size_ghost[2]; 
        ghost_region[i * 7 + 4].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 4].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 4].z_high = p_domain->local_sub_box_lattice_region.z_high;

        // 需要存储转发1次 先向前，再向右
        ghost_region[i * 7 + 5].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 5].y_low = p_domain->local_sub_box_lattice_region.y_low - p_domain->lattice_size_ghost[1]; // 0
        ghost_region[i * 7 + 5].z_low = p_domain->local_split_coord[2] - p_domain->lattice_size_ghost[2]; 
        ghost_region[i * 7 + 5].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 5].y_high = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 5].z_high = p_domain->local_sub_box_lattice_region.z_high;

        // 不需要存储转发 直接向右
        ghost_region[i * 7 + 6].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high); 
        ghost_region[i * 7 + 6].y_low = p_domain->local_sub_box_lattice_region.y_low; // 2
        ghost_region[i * 7 + 6].z_low = p_domain->local_split_coord[2] - p_domain->lattice_size_ghost[2]; 
        ghost_region[i * 7 + 6].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 6].y_high = p_domain->local_split_coord[1] + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 6].z_high = p_domain->local_sub_box_lattice_region.z_high;
        break;
      case 6:
        // 不需要存储转发 直接向上
        ghost_region[i * 7 + 0].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 0].y_low = p_domain->local_split_coord[1] - p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 0].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 0].x_high = 2 * (p_domain->local_split_coord[0] + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 0].y_high = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 0].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];
 
        // 需要存储转发1次 先向上，再向后
        ghost_region[i * 7 + 1].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 1].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 1].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 1].x_high = 2 * (p_domain->local_split_coord[0] + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 1].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 1].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 需要存储转发1次 先向上，再向左
        ghost_region[i * 7 + 2].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 2].y_low = p_domain->local_split_coord[1] - p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 2].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 2].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 2].y_high = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 2].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 需要存储转发2次 先向上，再向后，最后向左
        ghost_region[i * 7 + 3].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 3].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 3].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 3].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 3].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 3].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 不需要存储转发 直接向后
        ghost_region[i * 7 + 4].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 4].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 4].z_low = p_domain->local_split_coord[2] - p_domain->lattice_size_ghost[2]; 
        ghost_region[i * 7 + 4].x_high = 2 * (p_domain->local_split_coord[0] + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 4].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 4].z_high = p_domain->local_sub_box_lattice_region.z_high;

        // 需要存储转发1次 先向后，再向左
        ghost_region[i * 7 + 5].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 5].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 5].z_low = p_domain->local_split_coord[2] - p_domain->lattice_size_ghost[2]; 
        ghost_region[i * 7 + 5].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 5].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 5].z_high = p_domain->local_sub_box_lattice_region.z_high;

        // 不需要存储转发 直接向左
        ghost_region[i * 7 + 6].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_low - p_domain->lattice_size_ghost[0]); // 0
        ghost_region[i * 7 + 6].y_low = p_domain->local_split_coord[1] - p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 6].z_low = p_domain->local_split_coord[2] - p_domain->lattice_size_ghost[2]; 
        ghost_region[i * 7 + 6].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_low); // 2
        ghost_region[i * 7 + 6].y_high = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 6].z_high = p_domain->local_sub_box_lattice_region.z_high;
        break;
      case 7:
        // 不需要存储转发 直接向上
        ghost_region[i * 7 + 0].x_low = 2 * (p_domain->local_split_coord[0] - p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 0].y_low = p_domain->local_split_coord[1] - p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 0].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 0].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 0].y_high = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 0].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 需要存储转发1次 先向上，再向后
        ghost_region[i * 7 + 1].x_low = 2 * (p_domain->local_split_coord[0] - p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 1].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 1].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 1].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 1].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 1].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 需要存储转发1次 先向上，再向右
        ghost_region[i * 7 + 2].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 2].y_low = p_domain->local_split_coord[1] - p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 2].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 2].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 2].y_high = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 2].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 需要存储转发2次 先向上，再向后，最后向右
        ghost_region[i * 7 + 3].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 3].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 3].z_low = p_domain->local_sub_box_lattice_region.z_high;
        ghost_region[i * 7 + 3].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 3].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 3].z_high = p_domain->local_sub_box_lattice_region.z_high + p_domain->lattice_size_ghost[2];

        // 不需要存储转发 直接向后
        ghost_region[i * 7 + 4].x_low = 2 * (p_domain->local_split_coord[0] - p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 4].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 4].z_low = p_domain->local_split_coord[2] - p_domain->lattice_size_ghost[2]; 
        ghost_region[i * 7 + 4].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 4].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 4].z_high = p_domain->local_sub_box_lattice_region.z_high;

        // 需要存储转发1次 先向后，再向右
        ghost_region[i * 7 + 5].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high);
        ghost_region[i * 7 + 5].y_low = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 5].z_low = p_domain->local_split_coord[2] - p_domain->lattice_size_ghost[2]; 
        ghost_region[i * 7 + 5].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 5].y_high = p_domain->local_sub_box_lattice_region.y_high + p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 5].z_high = p_domain->local_sub_box_lattice_region.z_high;

        // 不需要存储转发 直接向右
        ghost_region[i * 7 + 6].x_low = 2 * (p_domain->local_sub_box_lattice_region.x_high); 
        ghost_region[i * 7 + 6].y_low = p_domain->local_split_coord[1] - p_domain->lattice_size_ghost[1];
        ghost_region[i * 7 + 6].z_low = p_domain->local_split_coord[2] - p_domain->lattice_size_ghost[2]; 
        ghost_region[i * 7 + 6].x_high = 2 * (p_domain->local_sub_box_lattice_region.x_high + p_domain->lattice_size_ghost[0]);
        ghost_region[i * 7 + 6].y_high = p_domain->local_sub_box_lattice_region.y_high;
        ghost_region[i * 7 + 6].z_high = p_domain->local_sub_box_lattice_region.z_high;
        break;
      default:
        assert(false);
        break;
    }
  }
}

void ABVIModel::addExchange_ghost(const _type_lattice_id& latticeId, const unsigned int sector_id){
  // ChangeLattice latti;
  // latti.type = lattice.type;
  // latti.x = lattice.id % box->lattice_list->meta.size_x;
  // latti.y = (lattice.id / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
  // latti.z = lattice.id / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);
  _type_lattice_id x = latticeId % box->lattice_list->meta.size_x;
  _type_lattice_id y = (latticeId / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
  _type_lattice_id z = latticeId / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);

  for(int i = 0; i < 7; i++) {
    if(ghost_region[sector_id * 7 + i].x_low <= x && x < ghost_region[sector_id * 7 + i].x_high && 
       ghost_region[sector_id * 7 + i].y_low <= y && y < ghost_region[sector_id * 7 + i].y_high && 
       ghost_region[sector_id * 7 + i].z_low <= z && z < ghost_region[sector_id * 7 + i].z_high) {

      if(!exchange_ghost[i].count(latticeId)) exchange_ghost[i].emplace(latticeId);
      // exchange_ghost[i].emplace_back(latti);
      break;
    }
  }
}

void ABVIModel::addExchange_surface(_type_lattice_id surface_id) {
  _type_lattice_id surface_x = surface_id % box->lattice_list->meta.size_x;
  _type_lattice_id surface_y = (surface_id / box->lattice_list->meta.size_x) % box->lattice_list->meta.size_y;
  _type_lattice_id surface_z = surface_id / (box->lattice_list->meta.size_x * box->lattice_list->meta.size_y);
  const int dims[comm::DIMENSION_SIZE] = {comm::DIM_X, comm::DIM_Y, comm::DIM_Z};
  // todo the regions can be static.
  std::array<std::vector<comm::Region<comm::_type_lattice_coord>>, comm::DIMENSION_SIZE> send_regions; // send regions in each dimension
  _type_lattice_size x_low;
  _type_lattice_size y_low;
  _type_lattice_size z_low;
  _type_lattice_size x_high;
  _type_lattice_size y_high;
  _type_lattice_size z_high;
  for (int sect = 0; sect < 8; sect++) {
    for (int d = 0; d < comm::DIMENSION_SIZE; d++) {
      send_regions[d] = comm::fwCommSectorSendRegion(sect, dims[d], p_domain->lattice_size_ghost,
                                                     p_domain->local_split_coord, p_domain->local_sub_box_lattice_region);
      for (auto &r : send_regions[d]) {
        x_low = 2 * r.x_low;
        y_low = r.y_low;
        z_low = r.z_low;
        x_high = 2 * r.x_high;
        y_high = r.y_high;
        z_high = r.z_high;
    
        if(x_low <= surface_x && surface_x < x_high && y_low <= surface_y && surface_y < y_high && z_low <= surface_z && surface_z < z_high){
          if(d == comm::DIM_X){
            auto it_from = exchange_surface_x[sect].find(surface_id);
            if(it_from == exchange_surface_x[sect].end()) exchange_surface_x[sect].emplace(surface_id);
          }else if(d == comm::DIM_Y) {
            auto it_from = exchange_surface_y[sect].find(surface_id);
            if(it_from == exchange_surface_y[sect].end()) exchange_surface_y[sect].emplace(surface_id);
          }else {
            auto it_from = exchange_surface_z[sect].find(surface_id);
            if(it_from == exchange_surface_z[sect].end()) exchange_surface_z[sect].emplace(surface_id);
          }
        }
      }
    }
  }
}

void ABVIModel::clear_exchange_ghost(){
  for (auto& vec : exchange_ghost) {
    vec.clear();
  }
}

void ABVIModel::clear_exchange_surface(const unsigned int next_sect){
  exchange_surface_x[next_sect].clear();
  exchange_surface_y[next_sect].clear();
  exchange_surface_z[next_sect].clear();
}

// std::array<std::vector<Lattice>, 7> ABVIModel::get_exghost(){
//   return exchange_ghost;
// }

// const comm::ColoredDomain* ABVIModel::get_pdomain(){
//   return p_domain;
// }

unsigned long ABVIModel::defectSize() {
  return 1; // todo add implementation
}