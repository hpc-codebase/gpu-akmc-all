---
name: HIP SKILL
description: AMD HIP 异构计算开发、CUDA 转 HIP、HIP 内核调试、HIP API 调用、性能优化
---

# ==================== HIP 编程 Skill 配置文件 ====================
# 适用场景：AMD HIP 异构计算开发、CUDA 转 HIP、HIP 内核调试、HIP API 调用、性能优化
# ==============================================================

[1. Skill 描述]
本技能在开发者进行 **AMD HIP 异构计算代码编写**、**CUDA 项目迁移至 HIP**、**HIP 内核函数开发/调试**、**HIP Runtime/加速库 API 使用**、**HIP 程序性能优化**、**HIP 代码语法/兼容性校验** 时自动触发；
核心能力：标准化输出可编译的 HIP 代码、CUDA-HIP 语法映射、问题排查方案、性能优化建议，完全适配 HIP 5.x+ 官方规范。

[2. 执行指令]
#codemap：HIP 异构计算编程专属指令集
## 命令
1. 严格遵循 AMD HIP 官方语法规范，兼容 HIP Runtime/hipBLAS/hipFFT/hipRand 等标准库；
2. CUDA 转 HIP 场景：自动完成语法映射（cudaMemcpy→hipMemcpy、cudaMalloc→hipMalloc、内核启动符<<<>>> 通用）；
3. 内核开发场景：严格区分 __global__（全局内核）、__device__（设备函数）、__host__（主机函数）修饰符；
4. 调试场景：精准定位内存错误、内核启动失败、数据传输异常、语法不兼容问题；
5. 输出要求：所有代码必须完整可编译，附带头文件、编译命令、关键注释；
6. 优化场景：提供 HIP 专属优化方案（共享内存、流并行、主机-设备数据传输优化）。

## 使用场景
1. 基础开发：HIP 主机端代码编写、设备内存管理、内核启动；
2. 代码迁移：CUDA 源码一键转换为可运行的 HIP 代码；
3. 内核开发：GPU 并行计算内核函数编写、调试；
4. 库调用：hipBLAS/hipFFT 等 AMD 加速库接口使用；
5. 问题排查：HIP 程序编译错误、运行时崩溃、性能瓶颈分析；
6. 工程化：HIP 项目 CMake 配置、编译脚本生成。

## 输出解释
1. 代码块：标准 C++/HIP 语法，可直接在 ROCm 环境中编译运行；
2. 注释：标注关键 API 作用、内存操作、内核参数；
3. 编译命令：提供 hipcc 编译指令（HIP 官方编译器）；
4. 解释说明：用简洁文字说明代码逻辑、API 用法、注意事项；
5. 示例输出：展示程序运行结果/内核执行效果。

## 示例
### 示例 1：HIP 基础向量加法（主机+设备代码）
```cpp
// 必需的 HIP 头文件
#include <hip/hip_runtime.h>
#include <iostream>

// HIP 设备内核函数：向量加法
__global__ void vectorAdd(const float* a, const float* b, float* c, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int size = 1024;
    const int bytes = size * sizeof(float);

    // 1. 主机端内存分配
    float *h_a = new float[size];
    float *h_b = new float[size];
    float *h_c = new float[size];

    // 2. 初始化数据
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = size - i;
    }

    // 3. 设备端内存分配
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    // 4. 数据从主机拷贝到设备
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    // 5. 启动 HIP 内核（线程块配置：256线程/块）
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);

    // 6. 结果从设备拷贝回主机
    hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);

    // 7. 验证结果
    std::cout << "HIP 向量加法执行完成！" << std::endl;
    std::cout << "索引 0 结果：" << h_c[0] << "（预期：1024）" << std::endl;

    // 8. 释放内存
    delete[] h_a; delete[] h_b; delete[] h_c;
    hipFree(d_a); hipFree(d_b); hipFree(d_c);

    return 0;
}