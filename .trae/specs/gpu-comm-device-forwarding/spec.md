# GPU 通信：接收侧存储转发 Device 化 Spec

## Why
当前 GPU 通信路径虽然已将 payload 传输与接收侧 apply 迁移到 RCCL/NCCL 与 device，但转发队列仍部分依赖 host `exchange_*`，导致 payload 仍需 D2H 回传用于 host 更新。需要把“接收后存储转发”与“增量来源”进一步迁移到 device buckets，逐步减少 host 参与并为关闭 D2H 做准备。

## What Changes
- 新增接收侧 device 存储转发：将接收 payload 中命中 ghost/surface 的 lattice 直接落桶到 device buckets
- 新增 vacancy event 增量的 device 落桶（非 recb 路径）
- 对齐清理点：每个 sector 开始时清空 device buckets（ghost 全清，surface 按 sect 清 3 个）
- **非目标**：本次不移除 host `exchange_*` 的维护，只在 `KMC_GPU_COMM=1` 下引入 device 转发能力并保持行为一致

## Impact
- Affected specs: GPU 通信迁移（Step 5/5.1/5.2）
- Affected code:
  - `gpu/gpu_simulate.h` / `gpu/gpu_simulate.hip`（device buckets、落桶与接收侧转发 kernel）
  - `src/algorithms/sl/sublattice.inl`（sector 清理点对齐、接收路径接入转发）
  - `ai_docs/gpu_comm_migration_steps.md`、`ai_docs/gpu_comm_change_log.md`（进度与变更记录）

## ADDED Requirements
### Requirement: Device Buckets Forwarding
系统 SHALL 在 `KMC_GPU_COMM=1` 时，将接收 payload 中需要继续转发/同步的 lattice 变更写入 device buckets。

#### Scenario: syncSimRegions 接收转发（Success）
- **WHEN** rank 在 `syncSimRegions` 的 GPU 接收路径完成 `transferBufferToGPU_Device`（apply + 坐标修正）
- **THEN** 系统将 `d_recv_payload`（除尾包 count）中命中 ghost/surface 的点落入 device buckets
- **AND** 后续发送侧可从 buckets compact 得到待发送 id 列表

#### Scenario: syncNextSectorGhostRegions 接收转发（Success）
- **WHEN** rank 在 `syncNextSectorGhostRegions` 的 GPU 接收路径完成 `transferBufferToGPU2_Device`
- **THEN** 系统将 `d_recv_payload` 全部元素参与落桶（该路径 payload 无尾包）

### Requirement: Vacancy Event Incremental Buckets
系统 SHALL 在 GPU vacancy event 执行完成后，将命中 ghost/surface 的增量直接写入 device buckets。

#### Scenario: GPU select+perform（Success）
- **WHEN** `selectAndPerformEventGPU` 完成一次 vacancy event
- **THEN** 若 `to_id` 属于 ghost，写入 sector 对应的 ghost buckets
- **AND** 若 `from_id/to_id` 属于 surface，写入 surface buckets

### Requirement: Buckets Clear Point Alignment
系统 SHALL 在每个 sector 开始前清理 device buckets，避免跨 sector 污染。

#### Scenario: Sector start（Success）
- **WHEN** sector loop 进入新 sector 并调用 host `clear_exchange_*`
- **THEN** 在 `KMC_GPU_COMM=1` 下同步清理 device buckets（ghost 全清；surface 清当前 sect 的 3 个 buckets）

## MODIFIED Requirements
### Requirement: Step 5/5.1/5.2 进度与说明
迁移步骤文档 SHALL 反映：方案 B 已覆盖 ghost/surface buckets、非 recb 增量来源、以及接收侧存储转发 device 化；并明确 host `exchange_*` 仍存在的原因与下一步关闭路径。

## REMOVED Requirements
无
