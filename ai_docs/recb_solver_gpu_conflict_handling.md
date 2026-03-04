# recb_solver：GPU 位置冲突处理方案（检测/解决/校验/记录）

## 1. 为什么必须做冲突处理

`calcExchangePairs` 会同时对大量候选原子执行“选随机邻点 → 根据类型规则进行交换/重组”。当多个线程同时触碰同一 lattice id（作为源或目标）时，会出现：
- 写冲突：同一个 hashset 槽位被多个线程插入/删除
- 语义冲突：同一 lattice id 在一次 step 中被多次交换，导致最终类型不可预测
- 触达点集合污染：ghost/surface keys 被重复写入，或被错误写入

当前实现用 `Busy_Set` 做互斥（插入 id 与 random_id），但在高并发/高装载因子下会产生大量探测与原子操作，且缺少冲突统计与可观测性。

## 2. 冲突检测机制设计

### 2.1 现状：Busy_Set（保留为兼容路径）
- 机制：开放定址哈希表，原子交换插入/删除。
- 关键 API：
  - `hash_set_contains`
  - `hash_set_insert_atom_op`
  - `hash_set_remove_atom_op`
  - 参考：[gpu_simulate.hip:L549-L699](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L549-L699)

问题：
- hash 探测长度与表装载因子强相关，冲突越多越慢。
- 错误/失败不可观测（缺少计数/日志）。

### 2.2 推荐：稠密锁（Dense Lock）作为主方案

新增 `d_lock`（device 常驻）：
- 类型：`uint32_t*`（或 `int*`）
- 大小：`meta._max_id`（若包含 ghost，可扩展到 `meta._max_id + ghost_pad` 或按 global id 映射）

锁语义：
- 0：未加锁
- 非 0：持锁标识（推荐使用 `tid+1` 或 `epoch<<bits | tid`，便于 debug）

基本原语：
- `try_lock(id, tag)`：`atomicCAS(&d_lock[id], 0, tag) == 0`
- `unlock(id, tag)`：`atomicCAS(&d_lock[id], tag, 0)`（避免误释放他人锁）

两把锁的死锁避免：
- 固定锁序：总是先锁 `min(id, random_id)`，再锁 `max(...)`。
- 若第二把失败：释放第一把并重试。

好处：
- O(1) 加锁路径，避免 Busy_Set 探测；
- 可直接输出冲突计数（锁失败次数）；
- 对百万级并发更稳定。

## 3. 冲突解决策略

### 3.1 第一阶段（推荐先落地）：锁 + 有界重试
- 每个候选最多尝试 `K` 次（例如 8/10）：
  1. 生成随机目标 `random_id`
  2. 尝试获取 `(id, random_id)` 两把锁
  3. 成功则执行交换逻辑并写触达点集合
  4. 失败则记录冲突计数并进入下一次尝试
- 若 `K` 次都失败：直接放弃该候选（或进入 retry_list，见 3.2）

### 3.2 可选增强：retry_list（二次 kernel 处理长尾）
目的：降低“高冲突区”导致的尾部延迟。

实现：
- device 维护：
  - `d_retry_list[N]`（保存 pair index 或 lattice id）
  - `d_retry_count`（原子计数）
- 主 kernel 中当重试次数耗尽时：
  - `idx = atomicAdd(d_retry_count, 1)`
  - `d_retry_list[idx] = pair_idx`
- 再 launch `calcExchangePairs_retry<<<>>>` 处理 retry_list（可用更保守的规则/更小并发）。

### 3.3 二阶段优化：空间分区（binning）
当冲突在空间上高度集中（例如大量缺陷聚簇）时，采用空间分区能显著降低冲突概率：
- 将局部 box 划分为固定 cell（如 8×8×8 或按经验选择）。
- 先按 cell 将 `Pair_Atoms` 分桶（bucketed list）。
- 分桶后 kernel 以 bucket 为单位处理，减少相邻线程对相同 id 的竞争。

注意：
- 当前工程未显式依赖 Thrust 的 sort/scan 算法调用；若不引入新依赖，可用“直方图 + prefix sum + scatter”自行实现分桶。

## 4. 校验与纠错逻辑

### 4.1 GPU 端轻量校验（建议仅在 debug/profile 开启）
- 交换前后类型一致性（membership 更新后 `getType(id)` 应符合预期分支）。
- hash_set insert/remove 返回值检查：
  - `insert` 失败（表满/探测失败）要计数并记录样本 id。
- ghost/surface 触达点合法性校验：
  - `isGhostLat/isSurfaceLat` 与 `meta.regions` 的坐标判断一致。

### 4.2 CPU 端回归校验
- 在小规模（例如 1e3~1e5）上，固定随机种子，对比 CPU 版本最终类型分布/关键统计量。
- 重点校验：
  - ghost/surface 交换集合是否一致（数量与具体 id）
  - `LatticesList` 中各集合（momo/more/re/vac）是否互斥且覆盖正确

## 5. 冲突事件记录（可观测性）

最低要求：
- `d_conflict_lock_fail`：锁失败次数
- `d_conflict_retry_push`：进入 retry_list 次数
- `d_hash_insert_fail`：hash_set 插入失败次数

可选：
- `d_conflict_log` 记录前 N 条（id、random_id、sector、loop_index、原因码），用于离线分析。

