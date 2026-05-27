---
layout: post
title: "Vector Addition: From Naive CUDA to H100-Optimized"
category: 技术
subcategory: CUTLASS
tags: [CUDA, Vector Addition, GPU Programming]
keywords: ["CUDA", "Vector Addition", "GPU Programming", "float4", "Performance Optimization", "H100"]
description: "A beginner-friendly guide to GPU vector addition in raw CUDA C++ — from a naive kernel to an H100-optimized implementation with vectorized memory access and grid-stride loops."
reference: "https://leetgpu.com/challenges/vector-addition"
---

# Vector Addition: From Naive CUDA to H100-Optimized

Vector addition is the "hello world" of GPU programming: `C[i] = A[i] + B[i]` for every element. It's trivially parallel — every element is independent — which makes it the perfect playground to learn how memory access patterns determine GPU performance.

## Background: How a GPU Executes Work

Before diving into code, let's establish the execution model:

- **Thread**: The smallest unit of execution. Each thread runs the same kernel code but on different data.
- **Block**: A group of threads (typically 128–1024) that can share fast on-chip memory and synchronize with each other.
- **Grid**: The collection of all blocks launched for a kernel. The grid covers the entire problem.
- **Warp**: A hardware scheduling unit of 32 threads within a block. All 32 threads in a warp execute the same instruction simultaneously (SIMT).

When you launch a kernel, the GPU maps your grid of blocks onto its Streaming Multiprocessors (SMs). Each SM runs one or more blocks concurrently.

## Understanding the H100 Memory System

The NVIDIA H100 (Hopper architecture) has a memory hierarchy designed for massive throughput:

| Level | Size | Bandwidth | Latency |
|-------|------|-----------|---------|
| HBM3 (Global Memory) | 80 GB | ~3.35 TB/s | ~400 cycles |
| L2 Cache | 50 MB | ~12 TB/s | ~200 cycles |
| L1 / Shared Memory | 256 KB per SM | ~33 TB/s (aggregate) | ~30 cycles |
| Registers | 256 KB per SM | Instant | 1 cycle |

Vector addition is **memory-bound** — the computation (one add per element) is trivial. The GPU spends almost all its time waiting for data from HBM3. Our job is to maximize memory bandwidth utilization.

## Part 1: The Naive Kernel

```cpp
__global__ void naive_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void solve(const float* A, const float* B, float* C, int N) {
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    naive_add_kernel<<<blocks, threads_per_block>>>(A, B, C, N);
}
```

### How It Works

1. **Index calculation**: Each thread gets a unique global index `idx = blockIdx.x * blockDim.x + threadIdx.x`.

2. **Bounds check**: `if (idx < N)` prevents out-of-bounds access when N isn't a multiple of the block size.

3. **Coalesced access**: Adjacent threads access adjacent memory addresses, so warps read contiguous memory — the hardware coalesces these into efficient 128-byte transactions.

### Why This Is Slow

The naive kernel typically achieves only 30–40% of peak HBM3 bandwidth on H100. The problem is **instruction overhead**: each thread issues one 4-byte load instruction per element. The memory controller coalesces warp-level accesses, but the instruction scheduler is overwhelmed by the sheer number of individual load/store instructions.

## Part 2: Vectorized with `float4`

### Key Insight: Vectorized Memory Access

Instead of each thread loading one 4-byte float, each thread can load **4 floats (16 bytes) in a single 128-bit instruction**:

```
Naive:      Thread 0 loads A[0]     → 4 bytes,  1 instruction
Vectorized: Thread 0 loads A[0:4]   → 16 bytes, 1 instruction (LDG.E.128)
```

Benefits:
- **4x fewer load instructions** → less instruction scheduler pressure
- **Better bus utilization** → each instruction carries more useful data
- **More in-flight bytes** → better latency hiding per Little's Law

### The Vectorized Kernel

```cpp
__global__ void vectorized_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_N = N / 4;

    if (idx < vec_N) {
        const float4* A4 = reinterpret_cast<const float4*>(A);
        const float4* B4 = reinterpret_cast<const float4*>(B);
        float4* C4 = reinterpret_cast<float4*>(C);

        float4 a = A4[idx];
        float4 b = B4[idx];

        C4[idx] = make_float4(
            a.x + b.x,
            a.y + b.y,
            a.z + b.z,
            a.w + b.w
        );
    }

    // Handle remaining elements (when N is not divisible by 4)
    int tail_start = vec_N * 4;
    int tail_idx = tail_start + threadIdx.x;
    if (threadIdx.x < (N - tail_start) && blockIdx.x == 0) {
        C[tail_idx] = A[tail_idx] + B[tail_idx];
    }
}

void solve(const float* A, const float* B, float* C, int N) {
    int threads_per_block = 256;
    int vec_N = N / 4;
    int blocks = (vec_N + threads_per_block - 1) / threads_per_block;
    vectorized_add_kernel<<<blocks, threads_per_block>>>(A, B, C, N);
}
```

### How `float4` Enables Vectorization

`float4` is a CUDA built-in type — a struct of four 32-bit floats, 16 bytes total, naturally aligned to 16 bytes. When you load a `float4`, the compiler emits a single `LDG.E.128` instruction that fetches all 16 bytes in one shot.

The `reinterpret_cast<const float4*>(A)` tells the compiler to treat the float array as an array of `float4`. Since `A[idx]` on a `float4*` accesses 4 contiguous floats, the compiler knows it can use a 128-bit load.

### What Is Memory Coalescing?

When 32 threads in a warp access addresses within the same 128-byte aligned region, the hardware serves all requests in a **single memory transaction**:

```
Good (coalesced):
  Thread 0 → address 0x1000    (4 bytes)
  Thread 1 → address 0x1004
  Thread 2 → address 0x1008
  ...
  → 1 transaction (128 bytes, fully utilized)

Bad (strided):
  Thread 0 → address 0x1000
  Thread 1 → address 0x2000
  → 32 separate transactions (mostly wasted bytes)
```

With `float4`, each thread accesses 16 contiguous bytes, and adjacent threads access adjacent `float4` elements. This means a warp issues 32 × 16 = 512 bytes of loads, served in 4 perfectly coalesced 128-byte transactions — with zero wasted bytes.

## Part 3: Grid-Stride Loop (Production Pattern)

For large N (25 million elements), we want each thread to process **multiple chunks** via a grid-stride loop. This reduces launch overhead and lets a smaller grid saturate the memory bus.

```cpp
__global__ void gridstride_add_kernel(const float* A, const float* B, float* C, int N) {
    int vec_N = N / 4;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    int stride = gridDim.x * blockDim.x;

    // Main vectorized loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < vec_N; idx += stride) {
        float4 a = A4[idx];
        float4 b = B4[idx];
        C4[idx] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    // Handle tail elements
    for (int idx = vec_N * 4 + blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += stride) {
        C[idx] = A[idx] + B[idx];
    }
}

void solve(const float* A, const float* B, float* C, int N) {
    int threads_per_block = 256;
    int blocks = min((N / 4 + threads_per_block - 1) / threads_per_block, 256);
    gridstride_add_kernel<<<blocks, threads_per_block>>>(A, B, C, N);
}
```

### Why Grid-Stride?

With N = 25,000,000 and `float4`, there are 6,250,000 vector elements to process. A grid of 256 blocks × 256 threads = 65,536 threads, so each thread handles ~95 `float4` chunks.

The grid-stride pattern `for (idx = global_tid; idx < vec_N; idx += stride)` lets every thread walk through the array in steps equal to the total grid width. Benefits:

- **Fixed grid size**: the number of blocks doesn't scale with N — you choose the grid size for optimal occupancy
- **Better cache behavior**: threads in a block stay close together in address space across iterations, improving L2 hit rate
- **Works for any N**: no need for a 1:1 thread-to-element mapping

### Capping Blocks at 256

The H100 has 132 SMs, each running up to ~16 blocks (depending on register/shared memory pressure). With 256 threads/block, 256 blocks is enough to fully occupy the GPU while keeping launch overhead minimal.

## Performance Expectations

| Kernel | Bandwidth (H100, float32) | % of Peak (3.35 TB/s) |
|--------|---------------------------|------------------------|
| Naive (scalar loads) | ~1.0–1.3 TB/s | ~30–40% |
| Vectorized (float4) | ~2.8–3.1 TB/s | ~85–93% |
| Grid-stride + float4 | ~2.9–3.2 TB/s | ~87–95% |
| Theoretical peak | 3.35 TB/s | 100% |

The remaining gap comes from kernel launch overhead, TLB misses on large tensors, and L2 bank conflicts.

## Summary

| Technique | What It Does | Why It Helps |
|-----------|-------------|--------------|
| `float4` loads/stores | Each thread loads 4 floats (16 bytes) per instruction | 4x fewer instructions vs scalar |
| `reinterpret_cast<float4*>` | Treats `float*` as `float4*` for 128-bit access | Required for compiler to emit `LDG.E.128` |
| Grid-stride loop | Each thread processes multiple chunks | Fixed grid size, better cache behavior, works for any N |
| Tail handling | Scalar loop for `N % 4` remaining elements | Correctness for non-divisible N |
| 256 threads/block | Good occupancy, multiple blocks per SM | Hides memory latency via warp switching |

## What's Next

Vector addition is memory-bound — the GPU barely computes anything. For compute-bound kernels (matrix multiply, convolution), the real optimization challenges appear: **tiled layouts**, **shared memory staging**, and **warp-specialized pipelines**. The vectorized access pattern you learned here — `float4` loads with grid-stride iteration — is a building block used in every high-performance CUDA kernel.
