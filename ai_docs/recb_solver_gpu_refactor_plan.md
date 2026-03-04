# recb_solver GPU(HIP) 代码重构：详细实现计划

## 0. 目标与边界

### 目标
- 将 `recb_solver` 的 GPU 部分从“可跑但偏串行/偏调试态”的实现，重构为面向百万级原子的高吞吐 HIP 实现。
- 保持与现有 CPU 侧数据结构与通信逻辑兼容：CPU 仍以 `LatticesList::setType()` 回写类型，并通过 `addExchange_ghost/addExchange_surface` 驱动后续 packer/MPI 同步。
- 通过结构性优化实现至少 5 倍性能提升（相对当前 CPU 版本的 `ABVIModel::recb_solver` 逻辑）。
- 达到“零编译警告、零运行时错误、全回归测试通过、文档/注释完整率 100%”。

### 边界/不变项
- 不改变现有 CMake 组织结构与 target 关系（仍然 `KMC_HIP_ENABLE_FLAG` → `add_subdirectory(gpu)`；`abvi` 链接 `gpu`）。
- 不改变 CPU 侧 `ABVIModel::recb_solver()` 的对外行为与接口签名（可在内部增加更合理的 buffer/预分配管理，但不破坏调用方）。
- 不引入新的外部依赖库（仅使用现有 HIP Runtime/hiprand）。

## 1. 现状梳理（代码定位与数据流）

### 1.1 CPU 调用链
- CPU 入口：`ABVIModel::recb_solver()`（负责计算规模、调用 GPU、回写 CPU 类型、标记 ghost/surface 交换集合）
  - 参考：[kmc.cpp:L258-L325](file:///Users/bytedance/Downloads/kmc_new/models/abvi/kmc.cpp#L258-L325)

### 1.2 GPU 实现入口与关键内核
- GPU 接口：`recb_solver_GPU(...)`
  - 参考：[gpu_simulate.hip:L1901-L1972](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1901-L1972)
- 核心内核：`calcExchangePairs(sector_id, arr_size)`
  - 参考：[gpu_simulate.hip:L1551-L1844](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1551-L1844)
- 类型回填内核：`get_surface_and_ghost_type(...)`（当前实现为 2 线程串行扫表）
  - 参考：[gpu_simulate.hip:L1857-L1900](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1857-L1900)
- 全局设备指针绑定：`setGlobalPointer(...)`，并通过 `__device__ getType()` 以“多哈希集合 membership”推断类型
  - 参考：[gpu_simulate.hip:L1129-L1169](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1129-L1169)

### 1.3 CPU-GPU 通信流程与“CPU 版本一致性”检查结论

#### 1.3.1 需要先澄清的事实
- 当前代码库中不存在单独的“CPU 版 calcExchangePairs/MoMo-MoRe 交换计算”实现；CPU 侧 `ABVIModel::recb_solver()` 只负责调用 GPU、回写 `LatticesList`、并标记 ghost/surface 交换集合。
  - 参考：[kmc.cpp:L264-L325](file:///Users/bytedance/Downloads/kmc_new/models/abvi/kmc.cpp#L264-L325)
- 因此所谓“与 CPU 版本保持一致”，在现状下等价于：CPU 对 GPU 的调用语义/参数假设、以及 GPU 回传的数据格式/含义，是否与 CPU 的回写逻辑匹配；并且 GPU 的类型空间/候选列表是否与 CPU 的类型系统一致。

#### 1.3.2 当前通信流程在逻辑上与 CPU 回写是匹配的部分（OK）
- CPU 对 `h_ghost_Hash/h_surface_Hash` 做了 `-1/-2` 过滤后才调用 `meta.isGhostLat/isSurfaceLat`，因此空槽不会触发错误回写。
  - 参考：[kmc.cpp:L280-L312](file:///Users/bytedance/Downloads/kmc_new/models/abvi/kmc.cpp#L280-L312)
- GPU 侧回传的 keys 与 type 数组按“同一槽位索引对齐”生成：`get_surface_and_ghost_type` 针对 `keys[i]` 写入 `type[i]`，CPU 侧对同一 `i` 使用 `setType(keys[i], type[i])`，对齐关系成立。
  - 参考：[gpu_simulate.hip:L1857-L1962](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1857-L1962)
- `sizes[0/1]` 当前回传的是 `ghost_Hash/surface_Hash` 的 capacity，CPU 侧用它作为循环上限与 memset 上限，语义一致（只是性能上会导致 O(capacity) 扫描与清空）。
  - 参考：[gpu_simulate.hip:L1967-L1968](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1967-L1968)、[kmc.cpp:L319-L322](file:///Users/bytedance/Downloads/kmc_new/models/abvi/kmc.cpp#L319-L322)

#### 1.3.3 当前存在的“不一致/潜在错误点”（需要在重构中修正）
- `recb_solver_GPU(sector_id, arr_size, ...)` 的 `arr_size` 入参在 GPU 侧没有被使用：实际 launch 使用的是全局 `pair_atoms_size`，这与 CPU 每次按 `momo_hash.size()+more_hash.size()` 动态计算规模的假设不一致。
  - 参考：[kmc.cpp:L268-L275](file:///Users/bytedance/Downloads/kmc_new/models/abvi/kmc.cpp#L268-L275)、[gpu_simulate.hip:L1914-L1924](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1914-L1924)
- 候选列表 `Pair_Atoms` 的构建与 CPU 语义不一致：`gpu_prepare` 中把 `momo_hash + more_hash + rere_hash` 合并成候选列表，但 GPU 侧 `getType()` 仅支持 `MoRe/MoMo/Re/V/Mo`，不包含 `ReRe`，会把 `rere_hash` 中的 id 视为 `Mo`。
  - 候选列表构建：[pkmc.cpp:L224-L237](file:///Users/bytedance/Downloads/kmc_new/profiles/pkmc.cpp#L224-L237)
  - GPU 类型判定（无 ReRe）：[gpu_simulate.hip:L1160-L1168](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1160-L1168)
  - CPU 类型系统含 ReRe：[lattices_list.cpp:L562-L572](file:///Users/bytedance/Downloads/kmc_new/src/lattice/lattices_list.cpp#L562-L572)
- 候选列表生命周期与“动态类型变化”不一致：CPU 每次 recb 都基于当前 `momo_hash/more_hash` 的 size 推导规模，但 GPU 的 `g_Pair_Atoms` 在 `gpu_prepare` 后基本不更新；`update_Pair_Atoms` 也被标注为有问题且不在主流程中调用。这会导致大量 stale id 被反复处理（浪费且可能影响统计/随机性路径）。
  - 参考：[gpu_simulate.hip:L1510-L1550](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1510-L1550)
- `pair_atoms_size = pair_atoms.size()*2` 与实际 host→device memcpy 数量不一致：device 端分配了 2 倍空间，但仅 memcpy 了 `pair_atoms.size()` 个元素，导致后一半为 `-1`，虽会在内核里被 `id <= 0 return` 过滤，但会造成至少 2 倍无效线程开销，并掩盖真实规模语义。
  - 参考：[gpu_simulate.hip:L2105-L2118](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L2105-L2118)、[gpu_simulate.hip:L1564-L1566](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1564-L1566)

### 1.4 当前主要性能/正确性风险点（必须重构）
- 内核内存在明显的“CPU 思维”串行代码路径：
  - `calcExchangePairs` 中每个线程扫描整张 `MoRe/MoMo` 表统计数量（`for i in capacity`），在百万级下会直接不可用。
  - 参考：[gpu_simulate.hip:L1614-L1623](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1614-L1623)
- `get_surface_and_ghost_type` 使用 `tid==0/1` 串行遍历 `capacity`，会成为 GPU→CPU 回传的硬瓶颈。
  - 参考：[gpu_simulate.hip:L1857-L1882](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1857-L1882)
- 每次调用 `recb_solver_GPU` 都在设备侧 `hipMalloc/hipFree` 分配 `ghost_type/surface_type`，频繁分配导致高开销与碎片风险。
  - 参考：[gpu_simulate.hip:L1945-L1971](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1945-L1971)
- `HIPHashSet::clear()` 当前实现每次分配临时 host 数组并 memcpy 到 device，极慢且会放大调用频次的成本。
  - 参考：[DeviceVacRatesSolver.cpp:L194-L222](file:///Users/bytedance/Downloads/kmc_new/gpu/DeviceVacRatesSolver.cpp#L194-L222)
- `gpu_final()` 中出现 `delete g_Busy_Set;` 这类对设备符号指针的错误释放路径，存在潜在崩溃/未定义行为风险。
  - 参考：[gpu_simulate.hip:L2169-L2202](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L2169-L2202)

#### 1.4.1 现象：交换后 MoMo/MoRe 数量“异常减少”的可能原因（排查要点）

该问题在当前实现里更可能来自“并发哈希集合更新不满足线性一致性”或“插入失败未回滚”，而不是来自交换规则本身。需要优先排查以下点：

- 并发插入实现存在覆盖风险（高概率根因）：
  - `hash_set_insert_atom_op` 在发现 `empty/tombstone` 后直接 `atomicExch(&keys[pos], key)`，缺少“期望值校验”的 CAS 语义；两个线程可能同时观察到空槽并依次 `atomicExch`，后写入者会覆盖先写入者，导致前一个 key 永久丢失，表现为集合元素数量下降。
  - 参考：[gpu_simulate.hip:L677-L699](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L677-L699)
- 64-bit 原子“读取”实现可能产生撕裂读（会放大表损坏概率）：
  - 当前 `atomicLoad(long*)` 通过两个 32-bit `atomicExch` 拼接 64-bit 值，读取过程中若有并发写入，可能读到高/低位来自不同时间片的组合值（torn value），导致错误判断为 `empty_flag/tombstone/某个 key`，进一步引起错误插入/删除或探测链损坏。
  - 参考：[gpu_simulate.hip:L640-L649](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L640-L649)
- 交换过程缺乏事务性：删除成功但插入失败不会回滚（会直接减少数量）：
  - 例如 MoMo↔Mo、MoRe↔Mo 的“位置交换”路径是“remove(old) + insert(new)”。一旦 `insert` 因表满/并发冲突失败（返回 0），当前代码不会把 old key 插回去，MoMo/MoRe 数量会减少。
  - 相关代码（remove/insert 连续执行）：[gpu_simulate.hip:L1709-L1712](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1709-L1712)、[gpu_simulate.hip:L1747-L1750](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1747-L1750)
- `Busy_Set` 只锁定 lattice id，不锁定哈希表探测链/槽位：
  - 即使 `id/random_id` 不冲突，不同 lattice id 仍可能落到相同哈希槽位或相邻探测区间，导致并发 insert/remove 在同一探测链上交错，进一步触发表损坏或 key 覆盖。
  - 参考 Busy_Set 使用处：[gpu_simulate.hip:L1644-L1695](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1644-L1695)
- 容量/装载因子问题会放大上述风险：
  - `HIPHashSet` 用开放定址线性探测实现，若 capacity 不足或 tombstone 过多，插入失败概率升高；当前实现对失败缺少可观测统计与恢复策略。
  - 参考构造与标记定义：[DeviceVacRatesSolver.h:L100-L125](file:///Users/bytedance/Downloads/kmc_new/gpu/DeviceVacRatesSolver.h#L100-L125)

对应排查建议（最小化改动即可定位）：
- 在设备端增加计数器（原子累加）：`insert_fail/remove_fail/overwrite_suspect`，区分 `MoMo_Hash/MoRe_Hash/Re_Hash` 三类集合。
- 对每条交换路径把 `remove/insert` 的返回值纳入分支：一旦 `insert` 失败，立即回滚插回旧 key，并记录一次失败事件（用于验证“事务性不足”是否是直接原因）。
- 将 `atomicExch` 改为 `atomicCAS(expected=empty/tombstone)` 的版本（先在 debug 分支验证是否还能复现“数量减少”），若现象消失，基本可确认并发插入覆盖为根因。

## 2. GPU 版本实现方案（HIP 核函数 + 并行策略）

### 2.1 总体策略（推荐：两阶段流水）
- 阶段 A（交换计算）：对 `Pair_Atoms` 中的候选原子并行执行“More/Momo 交换逻辑”，并生成：
  - `ghost_keys`：触达 ghost 的 lattice id（去重集合/或者 compact list）
  - `surface_keys`：触达 surface 的 lattice id（去重集合/或者 compact list）
- 阶段 B（类型计算与回传）：并行对 `ghost_keys/surface_keys` 计算类型，并批量回传 CPU。

该设计把“计算密集”和“回传/整理”分开，便于单独优化与计时，形成清晰性能模型。

### 2.2 数据并行映射
- 主并行维度：以“候选原子（pair/itl id）”为单位并行。
- 次并行维度（二维线程布局）：以“邻域索引 / 随机尝试索引”为第二维，减少单线程串行循环。
  - `threadIdx.x`：pair index（一个 block 内的候选原子）
  - `threadIdx.y`：neighbor lane（例如 0..7 对 1nn，或 0..13 对候选交换方向）
- 归约策略：对 `numSIA/numRe` 等统计采用 block 内 reduction（shared memory 或 warp shuffle），避免每个 pair 线程串行扫 `MAX_1NN`。

### 2.3 线程负载均衡
- 采用“固定尝试次数 + 早停”策略：每个候选最多尝试 `K` 次（例如 8/10），遇到条件不满足则提前退出。
- 对长尾线程：
  - 将尝试维度放入 `threadIdx.y` 并行化（或使用 warp-per-atom 模式），降低单线程迭代导致的尾部拖累。
  - 对冲突频繁的场景，额外增加“冲突重试队列”（可选）：将失败项写入 `retry_list`，二次 kernel 再处理。

## 3. GPU 内存布局与访问模式设计

### 3.1 类型表示与访问（两种方案）

#### 方案 A（小改动，风险低，推荐第一阶段先落地）
- 保留现有 `MoRe/MoMo/Re/V` 四张 `GPUHashSet` 的 membership 表达方式，继续使用 `getType(id)` 判定类型。
- 通过减少 `getType` 调用次数与提升局部性来获得性能：
  - 将 `get1nn` 的读取与 `getType` 查询合并为一次性批处理，减少重复查表。
  - 仅在必要时查询 `getType`（例如判断分支/生成回传类型时）。

#### 方案 B（结构升级，性能更高，推荐第二阶段评估）
- 新增稠密类型数组 `d_type[id]`（例如 `uint8_t`），由交换内核直接更新类型值：
  - `getType(id)` 退化为一次全局内存读取，彻底消除多哈希查表开销。
  - 仍保留哈希集合用于 CPU 侧集合一致性或 debug 校验（可选）。
- 代价：需要一次性分配 `meta._max_id(+ghost)` 大小数组，并在 host→device 初始化、device→host 校验上做额外工作。

本计划建议：先实施方案 A 达到“可用 + 5 倍”目标；若仍不够，再引入方案 B。

### 3.2 锁页内存（Pinned）复用
- 保持 `gpu_prepare()` 中对以下 host pinned buffer 的一次性分配与复用：
  - `lattice_list->h_ghost_Hash / h_surface_Hash`
  - `lattice_list->h_ghost_Hash_type / h_surface_Hash_type`
  - 参考：[gpu_simulate.hip:L2159-L2163](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L2159-L2163)
- 新增并长期复用 device 缓冲区（从每次 `hipMalloc/hipFree` 改为 prepare 阶段分配）：
  - `d_ghost_type / d_surface_type`
  - 可选：`d_retry_list / d_retry_count`（用于冲突重试队列）
  - 可选：`d_lock`（用于冲突处理的稠密锁，见第 5 节）

### 3.3 HashSet 清空策略（必须改）
- 将 `HIPHashSet::clear()` 从“host 临时数组 + memcpy”替换为：
  - `hipMemset(h_table.keys, 0xFF, capacity*sizeof(long int))`（-1 填充对 long int 等价）
  - 或使用专用 device kernel `clear_keys<<<>>>`（便于拓展到 tombstone 初始化）
- 目标：使每轮 `ghost_Hash/surface_Hash` 清空成本接近纯带宽上限。
  - 参考当前实现：[DeviceVacRatesSolver.cpp:L194-L222](file:///Users/bytedance/Downloads/kmc_new/gpu/DeviceVacRatesSolver.cpp#L194-L222)

## 4. 主机-设备通信流程与参数传递

### 4.1 接口设计（不改变现有签名，内部实现重构）
保持接口：
`recb_solver_GPU(sector_id, arr_size, h_ghost_Hash, h_surface_Hash, h_ghost_type, h_surface_type, sizes)`

内部变更：
- 取消每次调用的 device malloc/free（类型数组改为复用）。
- 取消每次调用中的多次 `hipDeviceSynchronize()`，改为：
  - kernel launch 后 `hipGetLastError` 检查
  - 仅在确实需要 host 读取结果时同步（回传 memcpy 隐含同步也可控）

### 4.2 参数传递方式
- 值传递：`sector_id、arr_size、meta` 的只读标量（或将 `meta` 常驻 device 并只更新必要字段）
- 指针传递：所有大数组与哈希表结构指针（keys、pair list、lock、output buffers）

### 4.3 数据传输频次与批量
- 每次 recb 仅回传必要的数据：
  - `ghost_keys/surface_keys`（key 数组）
  - `ghost_types/surface_types`（与 key 数组对齐）
  - `sizes`（capacity 或有效计数；建议升级为“有效计数”以减少 CPU 扫描）
- 优化方向：
  - 将“回传 keys 的 capacity 全量数组”升级为“compact 列表 + count”，避免 CPU 扫描大量 `-1/-2`。
  - 在 GPU 端通过前缀和/原子计数生成 compact list（适配百万级）。

## 5. 原子位置冲突处理（检测 + 解决 + 记录）

### 5.1 冲突来源
- 多个线程同时尝试修改同一个 lattice id（当前用 `Busy_Set` 做互斥）。
- 不同线程选择同一 `randomLattice.id` 作为目标，导致写冲突。
- 跨 sector / ghost 边界的交换会引入“逻辑上属于其它进程区域”的竞争。

### 5.2 冲突检测机制（推荐升级）

#### 现状（保留但要瘦身）
- `Busy_Set` 使用开放定址哈希表 + 原子交换实现插入/删除（正确性上可用，但冲突高时会很慢）。
  - 参考：`hash_set_insert_atom_op/remove_atom_op`：[gpu_simulate.hip:L653-L699](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L653-L699)

#### 推荐（新增稠密锁，作为主方案）
- 分配 `d_lock`（长度为 `meta._max_id(+ghost)`，元素类型 `uint32_t` 或 `int`）。
- 锁协议：
  - `try_lock(id)`：`atomicCAS(&d_lock[id], 0, tid_tag)` 成功则获得锁。
  - 交换需要锁定 `(id, random_id)` 两把锁：按较小 id 先锁，避免死锁。
  - 失败则释放已持有锁并进入下一次尝试（或写入 retry_list）。
- 优点：O(1) 互斥，不再依赖哈希探测；对百万级原子更稳定。

### 5.3 冲突解决算法（空间分区/分桶）
- 在“锁失败率较高”场景启用分桶：
  - 以 cell（例如 8x8x8）对空间分区，先按 cell 对 `Pair_Atoms` 做桶排序（可用 Thrust，但本项目目前没有引入 Thrust 的排序调用；也可自己实现 radix/binning）。
  - 同一个 bucket 内执行交换，跨 bucket 的竞争显著下降。
- 第一阶段可先不做排序，仅靠稠密锁与 retry_list 达到稳定；排序作为二阶段优化。

### 5.4 校验与纠错逻辑
- GPU 端增加轻量一致性校验（仅在 debug/测试构建开启）：
  - 交换后 `getType(id)` 前后一致性
  - hash_set 插入/删除返回值检查（失败时记录事件）
- 记录冲突事件：
  - 维护 `d_conflict_counter`（原子累加）
  - 可选记录前 N 条冲突样本到 `d_conflict_log`（用于离线分析）

## 6. 核函数线程配置（百万级支持）

### 6.1 动态网格/块大小
- 以候选原子数量 `N` 计算：
  - `block.x`：默认 256（根据设备 `prop.maxThreadsPerBlock` 自适应，可降到 128）
  - `block.y`：8（对应 1nn lanes）或 14（对应随机方向 lanes），视实现选择
  - `grid.x = ceil(N / block.x)`，`grid.y = 1`
- 若采用 warp-per-atom：
  - 一个 warp 处理一个候选原子，`grid = ceil(N / warps_per_block)`，`block = (32*warps_per_block)`

### 6.2 至少 10^6 原子
必须满足：
- `Pair_Atoms` 可容纳 ≥1e6 ids（建议 device buffer 使用 `int64`，并避免 `pair_atoms_size = size*2` 这类不一致行为）
- 任何 O(N * capacity) 的行为必须消除（尤其是 per-thread 扫全表）

## 7. 构建系统保持（仅做必要选项增量）

保持：
- 目录结构与 `add_subdirectory(gpu)` 不变。

仅建议增量（以“零警告”为目标，且只影响 HIP 编译单元）：
- 对 `gpu` target 增加更严格的 warning 设置，并修复对应代码告警：
  - 常见：未使用变量、类型不匹配、格式化字符串与类型不一致、signed/unsigned、`printf` 在 device 的限制等。
- 可选开启 `-O3` 与 `--offload-arch=`（由现有工程机制/环境变量决定，不在此计划强绑）。

## 8. 代码注释与待实现项标注（按验收要求落地）

需要明确标注：
- “空位交换核函数”为待实现/待校验状态（当前 `update_Pair_Atoms` 已存在 todo 且逻辑不完整，应明确标为待实现并隔离出 release 路径）。
  - 参考：[gpu_simulate.hip:L1510-L1550](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1510-L1550)
- 聚焦 `MoRe` 合金交换流程：将 `cur_type == MoRe` 分支拆成可测试的设备函数（便于单测与 perf profile）。
  - 参考：[gpu_simulate.hip:L1744-L1822](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1744-L1822)

说明：本项目当前文件存在大量调试 `printf` 与中英混杂注释。重构阶段会以“不新增无必要注释”为默认原则，仅按本条验收要求添加“接口说明/算法说明/待实现标注”三类注释，并尽量集中到关键入口与公共设备函数处。

## 9. 分阶段实施步骤（对应你的实现步骤拆解）

### 阶段 1：代码分析与基线建立
- 建立 perf baseline：
  - CPU `ABVIModel::recb_solver` 的 wall time
  - GPU `recb_solver_GPU` 的分段计时（kernel、clear、type compute、memcpy）
- 增加最小化 profile 开关（编译期宏或 runtime flag），避免默认输出 device printf。
- 输出基线报告（包含数据规模 N、GPU 型号、线程配置）。

### 阶段 2：GPU 核函数设计与重构（核心）
- 重写 `calcExchangePairs`：
  - 移除每线程扫全表统计与任何 device printf。
  - 引入二维线程布局/warp-per-atom，优化 1nn 统计与随机尝试。
  - 将 MoMo/MoRe 分支拆为 `__device__` 小函数，减少分支嵌套，提高可读性与可测试性。
- 重写 type 计算：
  - 用并行 kernel 替换 `get_surface_and_ghost_type` 的串行扫表。
  - 与回传流程组合为：`compute_types<<<>>>` → `hipMemcpyAsync`（可选 stream）。

### 阶段 3：内存管理系统重构
- 把 `recb_solver_GPU` 中 per-call 的 `hipMalloc/hipFree` 改为 `gpu_prepare` 一次性分配：
  - `d_ghost_type / d_surface_type`
- 优化 `HIPHashSet::clear()` 为 `hipMemset`（或 device kernel）实现。
- 修复 `gpu_prepare`/`gpu_final` 资源生命周期：
  - 明确 host/device 指针归属，纠正错误的 delete/free。

### 阶段 4：通信流程实现与参数传递优化
- 将回传结构从“capacity 全量 keys + CPU 扫描”升级为“compact list + count”（推荐）。
- 对 ghost/surface 触达点去重策略：
  - 若继续用 `ghost_Hash/surface_Hash`，则配合更快 clear 与更快 type compute。
  - 若改为 compact list，则用 `d_seen_bitmap` 或 `d_lock` 去重。

### 阶段 5：冲突处理开发
- 实施稠密锁 `d_lock`：
  - 替换/补充 Busy_Set 的互斥机制
  - 增加冲突计数与可选日志
- 引入 retry_list（可选）：
  - 把锁失败/规则失败的候选推迟到下一轮处理，降低尾部延迟。

### 阶段 6：测试验证
- 正确性：
  - 与 CPU 版本逐步对齐（小规模 deterministic seed）
  - ghost/surface 回传一致性（回传 keys 的去重与 type 正确）
- 性能：
  - N=1e5、5e5、1e6 的 scaling 曲线
  - 分段耗时占比（kernel vs clear vs memcpy）
- 稳定性：
  - 长时间运行无泄漏、无崩溃、无未定义行为

## 10. 验收标准落地方式（可量化）
- 性能 ≥5x：提供同一输入、同一物理参数下 CPU vs GPU 时间对比表（包含方差/多次运行平均）。
- 百万级原子：提供 N=1e6 的运行记录（含峰值显存、kernel 时间）。
- 零警告/零错误：在 CI/本地开启常规告警下编译通过（不以 `-w` 掩盖）。
- 回归测试：`tests/` 全部通过（包含 unit 与 unit-abvi）。
- 文档/注释：本目录下文档完整；关键接口处注释满足第 8 节要求。
