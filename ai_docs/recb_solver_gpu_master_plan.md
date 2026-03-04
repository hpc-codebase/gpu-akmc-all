# recb_solver GPU(HIP) 重构总计划（分步交付）

本计划把 `ai_docs/recb_solver_gpu_refactor_plan.md` 中的要求拆成可执行的阶段与里程碑，按“先止血修正确性 → 再重构性能结构 → 最后做百万级与5x验收”的顺序推进。

## 0. 输出物与约束（全程遵守）
- 不改 CMake 结构与 target 关系，仅做必要 HIP 编译选项增量（若必须）。
- CPU 侧 `ABVIModel::recb_solver()` 对外接口与语义不变（仍由 CPU 回写 type + 标记 ghost/surface）。
- 代码注释只做“接口说明/算法说明/待实现标注”三类，空位交换核函数明确标为待实现状态。

## 1. 阶段一：建立基线与可观测性（必须先做）

### 1.1 基线计时与规模记录
- 在 GPU 侧把 `recb_solver_GPU` 分段计时：exchange kernel、type compute、clear/reset、D2H memcpy。
- 记录每次调用的规模：候选数 N、ghost/surface capacity（以及有效 key 数的估计/统计）。

### 1.2 关掉默认调试输出
- 移除/禁用 `calcExchangePairs` 内的 device `printf`（保留可控开关）。

### 1.3 增加一致性与错误计数器（仅 debug/profile 开启）
- 统计：
  - `MoMo/MoRe/Re/V` 的 insert_fail/remove_fail
  - Busy/lock 冲突次数
  - 触达点收集数量（raw 与去重后）
- 目的：为后续“MoMo/MoRe 数量异常减少”提供可定位证据。

**交付物**
- 一份 baseline 报告（N=小/中规模各一次，输出分段耗时占比与计数器摘要）。

## 2. 阶段二：正确性止血（先修复“数量减少/表损坏”）

### 2.1 修复并发插入覆盖风险（高优先级）
- 将 `hash_set_insert_atom_op` 从 `atomicExch` 写入改为 CAS 语义：仅在观察到 `empty/tombstone` 时用 `atomicCAS` 抢占槽位。
- 对 64-bit 原子读写策略做一次整理（避免撕裂读路径引发表损坏）。

### 2.2 让交换操作具备事务性（必要回滚）
- 对所有 `remove(old)+insert(new)` 路径：
  - 若 `insert` 失败，必须把 `old` 插回（或保持原状）并计数。
- 对 `remove` 失败也计数并跳过该次交换（避免进一步污染）。

### 2.3 修正“候选列表/规模语义不一致”
- 明确 `arr_size` 与实际 kernel 使用的 N 一致（不要忽略入参，避免 stale 工作量）。
- 修正 `pair_atoms_size = size*2` 的 2 倍无效线程问题（让分配/拷贝/launch 三者一致）。
- 候选列表中 `ReRe` 的处理做决策：
  - 要么从候选中移除 `rere_hash`；
  - 要么补齐 GPU 类型系统对 ReRe 的表示与规则（更大改动）。

**交付物**
- “数量减少”现象可复现用例 + 修复后计数器证明（insert_fail 显著下降/归零，MoMo/MoRe 数量不再无故下降）。

## 3. 阶段三：通信与回传协议优化（降低 CPU 扫描与 D2H 带宽）

### 3.1 type 计算并行化
- 用并行 kernel 替换 `get_surface_and_ghost_type<<<1,2>>>` 的串行扫表。

### 3.2 从“capacity 全量槽位回传”升级为“compact list + count”
- GPU 端输出：
  - `ghost_list[0..ghost_count)`、`surface_list[0..surface_count)`
  - `ghost_type_list/surface_type_list` 与之对齐
- CPU 端只遍历 `count`，不再扫描大量 `-1/-2`。

### 3.3 pinned 与 device 缓冲复用
- 把 `recb_solver_GPU` 内 per-call 的 `hipMalloc/hipFree` 移至 `gpu_prepare` 常驻分配。
- 优化 `HIPHashSet::clear()` 为 `hipMemset` 或 device kernel。

**交付物**
- 新的回传接口实现（不改变对外签名的前提下，内部使用 count，或通过 sizes 传 count），并给出 D2H 传输量与耗时对比。

## 4. 阶段四：核心内核重构为百万级并行（性能主战场）

### 4.1 移除内核内全表扫描
- 删除 `calcExchangePairs` 中每线程扫描 `MoRe/MoMo` capacity 的统计逻辑。
- 若需要统计，用全局计数器或单独 kernel 计算，不放在主路径。

### 4.2 线程组织升级（二维/warp-per-atom）
- 采用二维线程布局并行化邻域统计与尝试循环，或 warp-per-atom 方案降低长尾。
- 固定尝试次数 K + 早停，确保可预测的吞吐。

### 4.3 冲突处理升级（推荐：稠密锁 + retry_list）
- 引入 `d_lock` 稠密锁，按 `min(id,random_id)`→`max` 锁序避免死锁。
- 可选 retry_list 二次处理，提高高冲突场景吞吐。

**交付物**
- N=1e6 下 kernel 可稳定执行，冲突计数可控，吞吐达到目标曲线（与阶段一 baseline 对比）。

## 5. 阶段五：测试与回归（验收闭环）

### 5.1 正确性对齐
- 小规模 deterministic case（固定随机种子策略），对齐：
  - 类型分布变化
  - ghost/surface 触达点集合

### 5.2 全回归与零告警
- 跑 `tests/unit` 与 `tests/unit-abvi`。
- 确保编译零 warning（不靠 `-w`）。

### 5.3 性能验收
- CPU vs GPU：同输入、同规模、重复多次取平均/方差。
- 提交性能报告：至少 5x，N=1e6 可跑。

**交付物**
- 验收清单逐项打勾（对应 `ai_docs/recb_solver_gpu_acceptance_checklist.md`）。

## 6. 推荐推进顺序（最短路径）
1) 阶段一（可观测）→ 2) 阶段二（止血修正确性）→ 3) 阶段三（回传与内存复用）→ 4) 阶段四（内核并行化/冲突）→ 5) 阶段五（回归+性能验收）

