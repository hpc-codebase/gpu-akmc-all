# CPU/GPU 进程间通信方案文档 Spec

## Why
当前仓库同时存在 CPU/MPI 与 GPU/RCCL(NCCL) 两套进程间通信路径，且伴随 device buckets、端到端 GPU 计算等改造后，通信的变量、方法、时间节点与边界变得复杂。需要形成一份统一、可审计、可用于排错与验证的“通信方案总览”文档。

## What Changes
- 新增一份面向工程排查的通信总览文档（Markdown）
  - CPU/MPI 通信方案：数据结构、发送/接收方法、触发时间节点、与模型数据结构的关系
  - GPU/RCCL(NCCL) 通信方案：device buffers、buckets、组包/收包、apply/forwarding、触发时间节点
  - 两套方案的异同点：数据源、队列语义、拷贝路径（H2D/D2H）、同步点、性能与风险点
- 文档必须列出通信用到的运行时变量/编译开关及其作用范围
- **BREAKING**：无（仅文档新增）

## Impact
- Affected specs: 通信迁移与端到端 GPU 路径的工程文档化
- Affected code: 无（仅新增文档，引用现有代码位置）
- Affected docs:
  - 计划新增：`ai_docs/cpu_gpu_comm_overview.md`（文件名可在实现阶段微调，但必须稳定、可长期维护）

## ADDED Requirements
### Requirement: 通信总览文档
系统 SHALL 提供一份文档，完整描述 CPU 与 GPU 两种进程间通信方案。

#### Scenario: 文档内容完整（Success）
- **WHEN** 开发者阅读该文档
- **THEN** 能明确回答：
  - 使用哪些变量/开关控制通信路径（运行时/编译期）
  - CPU/MPI 与 GPU/RCCL 的收发 API 与缓冲区来源
  - 通信发生在 time loop 的哪些节点（step/sector/维度循环）
  - 两种方案的队列语义与数据结构差异（exchange_* vs device buckets）
  - 当前已实现的端到端 GPU 范围与仍存在的边界/注意事项

### Requirement: 代码引用可追溯
文档 SHALL 为每个关键步骤提供至少一个可点击的代码引用链接，覆盖：
- `sublattice.inl` 中通信时间节点
- `gpu_comm.*` 中 RCCL/NCCL 收发实现
- packer（CPU 路径）与 device apply/forwarding（GPU 路径）的关键实现

## MODIFIED Requirements
无

## REMOVED Requirements
无

