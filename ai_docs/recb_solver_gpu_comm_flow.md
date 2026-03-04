# recb_solver：主机-设备通信流程与参数传递（现状 + 重构建议）

## 1. 现状通信链路（端到端）

### 1.1 初始化阶段（一次性）
- `PKMC::onCreate()`（profile 层）会调用 `gpu_prepare(...)`：
  - 构造/上传用于交换计算的 `Pair_Atoms`（device 常驻）
  - 构造/上传 `dev_meta`（device 常驻）
  - 构造 GPU 侧哈希集合（MoRe/MoMo/Re/V/Busy_Set + ghost/surface 收集表）
  - 分配 host pinned 的回传缓冲 `lattice_list->h_ghost_Hash/..._type` 等
  - 参考：[pkmc.cpp:L240-L244](file:///Users/bytedance/Downloads/kmc_new/profiles/pkmc.cpp#L240-L244)、[gpu_simulate.hip:L2028-L2167](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L2028-L2167)

### 1.2 每次 recb 计算（反复调用）
- CPU 侧：
  - `ABVIModel::recb_solver()` 计算 `arr_size` 并调用 `recb_solver_GPU(...)`
  - 参考：[kmc.cpp:L268-L275](file:///Users/bytedance/Downloads/kmc_new/models/abvi/kmc.cpp#L268-L275)

- GPU 侧（`recb_solver_GPU`）：
  1. launch `calcExchangePairs(sector_id, pair_atoms_size)` 计算交换与触达点收集
  2. 计算触达点类型（现状为 `get_surface_and_ghost_type` 串行扫表）
  3. memcpy 回传：
     - `ghost_Hash->keys` → `h_ghost_Hash`
     - `surface_Hash->keys` → `h_surface_Hash`
     - `ghost_type/surface_type` → `h_ghost_Hash_type/h_surface_Hash_type`
  4. 清空 `ghost_Hash/surface_Hash`，等待下一轮
  - 参考：[gpu_simulate.hip:L1901-L1972](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1901-L1972)

- CPU 回写与“通信标记”：
  - 遍历 `h_ghost_Hash`：属于 ghost 的 id 调 `addExchange_ghost(id, sector_id)`，并 `setType(id, type)` 更新 CPU 集合
  - 遍历 `h_surface_Hash`：属于 surface 的 id 调 `addExchange_surface(id)`，并 `setType(id, type)`
  - 清空回传缓冲（memset -1）
  - 参考：[kmc.cpp:L279-L322](file:///Users/bytedance/Downloads/kmc_new/models/abvi/kmc.cpp#L279-L322)

## 2. 参数传递方式（现状）

`recb_solver_GPU` 的参数：
- 值传递：
  - `sector_id`：当前 sector（0..7）
  - `arr_size`：CPU 计算的规模（现状在 GPU 内并未严格使用，使用了 `pair_atoms_size` 全局）
- 指针传递（host 侧）：
  - `h_ghost_Hash/h_surface_Hash`：host pinned keys 数组指针（由 `gpu_prepare()` 分配）
  - `h_ghost_type/h_surface_type`：host pinned type 数组指针（由 `gpu_prepare()` 分配）
  - `sizes`：输出容量（capacity）给 CPU 用于循环上限

设备侧通过 `setGlobalPointer` 绑定常驻指针：
- `g_Pair_Atoms`、`g_meta`、四类 membership hashset、busy_set、ghost/surface hashset
  - 参考：[gpu_simulate.hip:L1129-L1153](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1129-L1153)

## 3. 重构建议：降低传输频次与 CPU 扫描成本

### 3.1 从“capacity 全量”升级为“compact list + count”

现状问题：
- keys 数组是 hashset 的槽位数组，容量为 `2*size`，回传时必须 memcpy 全量槽位；
- CPU 端必须扫描大量 `-1/-2` 空槽才能找出有效 key。

建议升级协议：
- GPU 端产生：
  - `ghost_list[0..ghost_count)`、`surface_list[0..surface_count)`
  - `ghost_type_list[...]` 与之对齐
- CPU 端只循环有效计数，避免 O(capacity) 扫描。

实现路径（与现有结构兼容）：
- 继续使用 `ghost_Hash/surface_Hash` 做去重，但在回传前做一次 `compact_keys<<<>>>`：
  - 输入：`keys[capacity]`
  - 输出：`list[count]`（通过原子加写入）
- 或者直接用 `d_seen`/`d_lock` 做去重并写 list（跳过 hashset 槽位回传）。

### 3.2 数据传输批量化与同步优化
- 将 type 计算与 keys 回传合并为两个并行 kernel：
  - `compute_type_for_keys<<<>>>`（并行填 type）
  - `compact_and_copy<<<>>>`（可选）
- 减少显式 `hipDeviceSynchronize()`：
  - 使用 `hipMemcpyAsync` + stream（如果代码库当前已使用 stream，则复用；否则先保持默认 stream 并减少同步点）

## 4. 与 MPI/packer 的衔接点（需要保持的语义）

`ABVIModel::recb_solver` 回写后生成的集合会被后续 packer 使用（ghost/surface 同步）。
建议在重构验证阶段，把以下链路作为回归重点：
- `addExchange_ghost/addExchange_surface` 触发的数据最终是否被 `ghost_sync_packer / sim_sync_packer` 正确打包并广播。
  - packer 目录参考：[ghost_sync_packer.cpp](file:///Users/bytedance/Downloads/kmc_new/src/pack/ghost_sync_packer.cpp)、[sim_sync_packer.cpp](file:///Users/bytedance/Downloads/kmc_new/src/pack/sim_sync_packer.cpp)

