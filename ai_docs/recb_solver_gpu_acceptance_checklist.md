# recb_solver GPU(HIP) 重构：验收清单（Checklist）

## 1. 功能正确性
- [ ] `ABVIModel::recb_solver()` 行为与接口保持不变（外部调用无需改动）
  - 参考：[kmc.cpp:L258-L325](file:///Users/bytedance/Downloads/kmc_new/models/abvi/kmc.cpp#L258-L325)
- [ ] GPU 交换逻辑覆盖 MoMo 与 MoRe 分支，且与 CPU 期望语义一致
  - 参考现状内核：[gpu_simulate.hip:L1551-L1844](file:///Users/bytedance/Downloads/kmc_new/gpu/gpu_simulate.hip#L1551-L1844)
- [ ] ghost/surface 触达点回传正确：
  - [ ] keys 去重正确（同一 id 不会导致重复回写/重复通信标记）
  - [ ] types 与 keys 对齐且类型正确（`setType(id, type)` 后 CPU 集合一致）
- [ ] `gpu_prepare()` → 多次 `recb_solver_GPU()` → `gpu_final()` 生命周期内无泄漏、无 double-free、无未定义行为

## 2. 性能目标（至少 5 倍）
- [ ] 提供 CPU vs GPU 的对比数据（同输入、同随机种子策略、同规模）
- [ ] N=1e6 规模下 GPU 端耗时可拆分为：
  - [ ] exchange kernel
  - [ ] type compute
  - [ ] clear/reset
  - [ ] memcpy（H2D/D2H）
- [ ] 关键串行瓶颈已消除：
  - [ ] 内核内不存在 per-thread 扫全 hash table 的循环
  - [ ] type 计算不再由 1~2 个线程串行遍历 capacity
  - [ ] recb_solver 每轮不再 `hipMalloc/hipFree` 临时数组
  - [ ] `HIPHashSet::clear()` 不再做 host 临时数组分配

## 3. 百万级原子支持
- [ ] `Pair_Atoms`/候选列表能容纳 ≥ 1e6 ids，且 kernel 网格计算不会溢出
- [ ] 线程配置支持动态计算 grid/block（不硬编码只适配小规模）
- [ ] 输出缓冲能处理百万级触达点（或采用 compact list + count 避免 capacity 过大）

## 4. 冲突处理
- [ ] 实现冲突检测（至少具备锁失败计数或 Busy_Set 冲突计数）
- [ ] 实现冲突解决策略（锁 + 重试 或 空间分区）
- [ ] 在高冲突场景下不会死锁、不会导致数据结构损坏

## 5. 构建系统与告警
- [ ] CMake 结构与 target 关系保持不变（仅增量必要选项）
  - 参考：[CMakeLists.txt](file:///Users/bytedance/Downloads/kmc_new/CMakeLists.txt)、[gpu/CMakeLists.txt](file:///Users/bytedance/Downloads/kmc_new/gpu/CMakeLists.txt)
- [ ] 编译零警告（至少在项目当前默认编译选项下为 0）
- [ ] 运行无 HIP runtime error（kernel launch/非法访问/未初始化）

## 6. 测试与回归
- [ ] 通过 `tests/unit` 回归
- [ ] 通过 `tests/unit-abvi` 回归
- [ ] 增加（或补齐）与 recb_solver/GPU 相关的最小可重复测试用例：
  - [ ] 小规模 deterministic case（便于 CPU/GPU 结果对齐）
  - [ ] 中规模压力 case（验证稳定性与吞吐）

## 7. 文档与注释（100% 完整率）
- [ ] 关键文档已输出到 `ai_docs/`：
  - [ ] 详细实现计划：[recb_solver_gpu_refactor_plan.md](file:///Users/bytedance/Downloads/kmc_new/ai_docs/recb_solver_gpu_refactor_plan.md)
  - [ ] 变量清单与生命周期：[recb_solver_gpu_variables.md](file:///Users/bytedance/Downloads/kmc_new/ai_docs/recb_solver_gpu_variables.md)
  - [ ] 通信流程：[recb_solver_gpu_comm_flow.md](file:///Users/bytedance/Downloads/kmc_new/ai_docs/recb_solver_gpu_comm_flow.md)
  - [ ] 冲突处理方案：[recb_solver_gpu_conflict_handling.md](file:///Users/bytedance/Downloads/kmc_new/ai_docs/recb_solver_gpu_conflict_handling.md)
- [ ] 代码注释规范要求落地：
  - [ ] 明确标注“空位交换核函数”为待实现状态
  - [ ] `MoRe` 合金交换流程的接口说明清晰
  - [ ] 保留原有功能标记与版本信息

