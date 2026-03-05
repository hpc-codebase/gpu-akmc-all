# Tasks
- [x] Task 1: 汇总当前 GPU 通信迁移进度与边界
  - [x] 从关键代码路径提炼“已完成/未完成”清单
  - [x] 明确当前仍需 D2H 的原因与触发点
- [x] Task 2: 实现接收侧存储转发 device 化
  - [x] 新增 `commForwardFromRecvPayload` 与 kernel（ghost/surface 落桶）
  - [x] 在 `syncSimRegions/syncNextSectorGhostRegions` 的 GPU 接收路径接入
- [x] Task 3: 覆盖非 recb 增量来源与清理点
  - [x] 在 `selectAndPerformEventGPU` 写入 device buckets
  - [x] 在 sector loop 开始处对齐清理（ghost 全清、surface 按 sect 清理）
- [x] Task 4: 更新文档
  - [x] 更新迁移步骤（Step 5/5.1/5.2）
  - [x] 更新变更记录（新增对应变更条目）

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 2 and Task 3
