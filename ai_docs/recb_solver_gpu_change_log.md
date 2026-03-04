# recb_solver GPU(HIP) 修改记录（逐步重构）

## 变更 001：加入 recb_solver_GPU 分段计时开关 + 默认禁用 device printf

### 修改文件
- [gpu_simulate.hip](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip)

### 修改内容
- 增加基于环境变量 `KMC_RECB_PROFILE` 的分段计时输出：
  - `exchange_ms`：`calcExchangePairs` kernel（含同步）
  - `type_ms`：触达点 type 计算 kernel（含同步）
  - `copy_ms`：D2H memcpy（keys/types）
  - `clear_ms`：`ghost_Hash/surface_Hash` 清空
- 在每次 kernel launch 后增加 `CHECK_FOR_ERROR()`，便于第一时间捕获 kernel launch 失败。
- 默认关闭 `calcExchangePairs` 内的 `printf("begin calcExchangePairs")`：
  - 新增编译宏 `KMC_RECB_DEVICE_PRINTF`（默认 0），只有手动打开时才输出。
- 在 `gpu_final()` 中销毁用于计时的 hipEvent，避免资源泄漏。

### 为什么要这样改
- 你当前的重构目标包含“性能提升 5 倍、百万级规模、零运行时错误”，必须先把性能分解成可观测的几段，否则无法判断瓶颈是否来自：
  - 主交换内核本体
  - type 计算串行化
  - 回传 memcpy 带宽
  - hashset clear/reset
- device `printf` 会显著改变 kernel 的执行时序与吞吐（甚至影响并发竞态的表现），作为默认路径会干扰“MoMo/MoRe 数量减少”等问题的定位，因此必须默认关闭，只保留可控开关。

### 对行为/接口的影响
- 不改变任何对外函数签名。
- 默认不输出 profile 信息；仅当设置 `KMC_RECB_PROFILE=1` 时在 stdout 输出一行统计。
- 默认不输出 device printf；仅当编译时定义 `KMC_RECB_DEVICE_PRINTF=1` 才输出。

### 如何使用
- 开启 profile（运行时）：
  - `export KMC_RECB_PROFILE=1`
- 开启 device printf（编译期）：
  - 需要在 HIP 编译单元定义 `KMC_RECB_DEVICE_PRINTF=1`（本轮未改 CMake，后续若需要可单独补充）。

## 变更 002：修复 GPUHashSet 并发插入/删除竞态 + 交换路径失败回滚

### 修改文件
- [gpu_simulate.hip](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip)

### 修改内容
- 将 `hash_set_insert_atom_op` 的“空槽写入”从 `atomicExch` 改为 `atomicCAS` 期望值更新：
  - 仅当槽位值仍为 `empty_flag/tombstone_flag` 时才写入 key，避免多个线程同时插入导致后写覆盖前写（key 丢失 → MoMo/MoRe 数量减少）。
- 将 `hash_set_remove_atom_op` 的“删除写入 tombstone”从 `atomicExch` 改为 `atomicCAS(expected=key)`：
  - 仅当槽位仍为待删除 key 时才置 tombstone，避免误删其它线程刚写入的 key。
- 为 `calcExchangePairs` 中 MoMo/MoRe 的关键交换路径补齐失败处理与回滚：
  - 对 `remove(old) + insert(new)` 型“位置交换”，若 `insert` 失败则把 `old` 插回。
  - 对 `MoMo+Re -> MoRe`、`MoRe<->Re` 等多步更新，任一步失败则尽可能恢复到交换前状态并跳过本次交换。

### 为什么要这样改
- 你反馈的“交换后 MoMo/MoRe 减少”更符合并发数据结构被破坏或事务不完整导致的元素丢失，而不是交换规则本身必然导致减少：
  - `atomicExch` 在空槽竞争下天然可能覆盖写，造成先插入的 key 被覆盖丢失。
  - `remove` 若不用 CAS 校验，可能把非目标 key 写成 tombstone。
  - 交换路径缺少回滚时，`remove` 成功但 `insert` 失败会直接减少集合数量。
- 本轮改动的目标是先“止血”，把集合数量异常减少从“常态”变成“可计数的失败事件”，为后续性能重构提供稳定基础。

### 对行为/接口的影响
- 不改变任何对外接口。
- 交换失败会更倾向于“跳过本次交换”（并释放 busy），以保证集合不被破坏。

