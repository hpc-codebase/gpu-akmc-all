# recb_solver GPU(HIP) 变量清单与生命周期（主机/设备）

本文聚焦 `recb_solver_GPU/calcExchangePairs` 相关变量，明确类型、大小、初始化位置、驻留位置与生命周期，用于后续“锁页内存复用、零泄漏、零未定义行为”的重构落地。

## 1. 设备侧全局符号（__device__ static）

定义位置：[gpu_simulate.hip:L6-L16](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L6-L16)

| 变量 | 类型 | 位置 | 初始化/绑定位置 | 生命周期 | 用途 |
|---|---|---|---|---|---|
| `g_MoRe_Hash` | `GPUHashSet*` | device 全局 | `setGlobalPointer(...)` | `gpu_prepare()` 后到 `gpu_final()` 前 | MoRe membership 表 |
| `g_MoMo_Hash` | `GPUHashSet*` | device 全局 | 同上 | 同上 | MoMo membership 表 |
| `g_Re_Hash` | `GPUHashSet*` | device 全局 | 同上 | 同上 | Re membership 表 |
| `g_V_Hash` | `GPUHashSet*` | device 全局 | 同上 | 同上 | Vacancy membership 表 |
| `g_Busy_Set` | `GPUHashSet*` | device 全局 | 同上 | 同上 | 并发互斥/去重（现状） |
| `g_surface_Hash` | `GPUHashSet*` | device 全局 | 同上 | 同上 | surface 触达点收集 |
| `g_ghost_Hash` | `GPUHashSet*` | device 全局 | 同上 | 同上 | ghost 触达点收集 |
| `g_meta` | `dev_meta*` | device 全局 | 同上 | 同上 | 区域边界、box/ghost 参数 |
| `g_Pair_Atoms` | `long int*` | device 全局 | 同上 | 同上 | 候选原子列表（pair atoms） |

备注：
- `setGlobalPointer` 绑定见：[gpu_simulate.hip:L1129-L1153](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1129-L1153)

## 2. HIPHashSet/GPUHashSet 结构（host 管理，device 数据驻留）

定义位置：[DeviceVacRatesSolver.h:L100-L125](file:///Users/bytedance/Downloads/kmc_new/gpu/DeviceVacRatesSolver.h#L100-L125)

| 名称 | 类型 | 位置 | keys 数组位置 | 容量 | 初始化位置 | 释放位置 |
|---|---|---|---|---|---|---|
| `GPUHashSet` | `struct` | device（被 `hipMalloc` 出来并拷贝） | `long int* keys`（device） | `int capacity` | `HIPHashSet` 构造函数 | `HIPHashSet` 析构 |
| `HIPHashSet` | `class` | host（new/delete） | 内部 `h_table.keys` 指向 device keys | `h_table.capacity` | `gpu_prepare()` 中构造 | `gpu_final()` 中 delete |

重要实现细节：
- `HIPHashSet::clear()` 当前会分配临时 host 数组并 `hipMemcpy` 到 device：[DeviceVacRatesSolver.cpp:L194-L222](file:///Users/bytedance/Downloads/kmc_new/gpu/DeviceVacRatesSolver.cpp#L194-L222)
  - 重构目标：改成 `hipMemset(keys, 0xFF, bytes)` 或 device kernel，避免 per-call host 分配。

## 3. recb_solver 相关 host/device 变量（gpuSimulate 命名空间）

定义位置：[gpu_simulate.hip:L18-L71](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L18-L71)

### 3.1 host 侧常驻指针（由 gpu_prepare 初始化，gpu_final 释放）

| 变量 | 类型 | 位置 | 初始化位置 | 释放位置 | 说明 |
|---|---|---|---|---|---|
| `lattice_list` | `LatticesList*` | host | `gpu_prepare()` | 不释放（外部所有） | CPU lattice 容器 |
| `p_domain` | `comm::ColoredDomain*` | host | `gpu_prepare()` | 不释放（外部所有） | domain/sector 信息 |
| `meta` | `dev_meta` | host value | `initGpuInfo_exchange` | 自动释放 | device meta 的 host mirror |
| `MoRe_Hash/MoMo_Hash/Re_Hash/V_Hash/Busy_Set` | `HIPHashSet*` | host(new) | `initGpuInfo_exchange` 内或 `gpu_prepare` | `gpu_final()` delete | GPU membership/互斥集合 |
| `ghost_Hash/surface_Hash` | `HIPHashSet*` | host(new) | `gpu_prepare()` | `gpu_final()` delete | 触达点集合 |

### 3.2 device 侧常驻指针（由 gpu_prepare 分配，gpu_final 释放）

| 变量 | 类型 | 位置 | 初始化位置 | 释放位置 | 说明 |
|---|---|---|---|---|---|
| `d_meta` | `dev_meta*` | device | `gpu_prepare()` hipMalloc+Memcpy | `gpu_final()` hipFree | `g_meta` 指向它 |
| `d_Pair_Atoms` | `long int*` | device | `gpu_prepare()` hipMalloc+Memcpy | `gpu_final()` hipFree | `g_Pair_Atoms` 指向它 |

### 3.3 host pinned（锁页）回传缓冲（由 gpu_prepare 分配，建议 gpu_final 释放）

分配位置：[gpu_simulate.hip:L2159-L2163](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L2159-L2163)

| 变量 | 类型 | 位置 | 规模 | 用途 |
|---|---|---|---|---|
| `lattice_list->h_ghost_Hash` | `long int*` | host pinned | `ghost_Hash->capacity` | 回传 ghost keys（当前为 capacity 全量） |
| `lattice_list->h_surface_Hash` | `long int*` | host pinned | `surface_Hash->capacity` | 回传 surface keys |
| `lattice_list->h_ghost_Hash_type` | `long int*` | host pinned | `ghost_Hash->capacity` | 回传 ghost types（与 keys 对齐） |
| `lattice_list->h_surface_Hash_type` | `long int*` | host pinned | `surface_Hash->capacity` | 回传 surface types |

重构建议：
- 在 `gpu_final()` 中补齐上述 pinned buffer 的 `hipHostFree`，并确保与 `LatticesList` 生命周期一致（避免 double-free）。
- 回传协议建议从“capacity 全量数组”升级为“compact list + count”，这样 host pinned buffer 规模为“最大可能有效数”而非 `capacity`。

## 4. 每次调用 recb_solver_GPU 的临时变量（需要消除的热点）

当前 `recb_solver_GPU()` 每次会分配 device 临时类型数组：
- `ghost_type/surface_type`（device malloc/free）
  - 参考：[gpu_simulate.hip:L1945-L1971](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1945-L1971)

重构目标：
- 将其提升为 `gpu_prepare()` 一次性分配并复用（例如 `d_ghost_type/d_surface_type`），每轮仅 `hipMemset` 或 `kernel` 重置必要范围。

## 5. 待新增变量（用于冲突处理与回传压缩）

| 变量 | 类型 | 位置 | 初始化位置 | 生命周期 | 用途 |
|---|---|---|---|---|---|
| `d_lock` | `uint32_t*` / `int*` | device | `gpu_prepare()` | 常驻 | 稠密锁，替代/补充 `Busy_Set` |
| `d_ghost_list` / `d_surface_list` | `long int*` | device | `gpu_prepare()` | 常驻 | compact list 存储回传 keys |
| `d_ghost_count` / `d_surface_count` | `int*` | device | `gpu_prepare()` | 常驻 | list 有效计数（原子加） |
| `d_conflict_counter` | `uint64_t*` | device | `gpu_prepare()` | 常驻 | 冲突统计 |

