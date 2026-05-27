---
layout: post
title: "Use CuTe to Implement Vector Addition"
category: 技术
subcategory: CUTLASS
tags: [CUTE, Vector Addition, GPU Programming]
keywords: ["CUTE", "Vector Addition", "GPU Programming", "CUDA", "Performance Optimization", "H100"]
description: "A beginner-friendly guide to GPU vector addition using the CuTe DSL — from a naive kernel to an H100-optimized implementation with vectorized memory access."
reference: "https://leetgpu.com/challenges/vector-addition"
---

# Vector Addition with CuTe: From Naive to H100-Optimized

Vector addition is the "hello world" of GPU programming: C[i] = A[i] + B[i] for every element. It's trivially parallel — every element is independent — which makes it the perfect playground to learn how memory access patterns determine GPU performance.

## Background: How a GPU Executes Work

Before diving into code, let's establish the execution model:

- **Thread**: The smallest unit of execution. Each thread runs the same kernel code but on different data.
- **Block**: A group of threads (typically 128–1024) that can share fast on-chip memory and synchronize with each other.
- **Grid**: The collection of all blocks launched for a kernel. The grid covers the entire problem.
- **Warp**: A hardware scheduling unit of 32 threads within a block. All 32 threads in a warp execute the same instruction simultaneously (SIMT).

When you launch a kernel, the GPU maps your grid of blocks onto its Streaming Multiprocessors (SMs). Each SM runs one or more blocks concurrently.

## Part 1: The Naive Kernel

```python
@cute.kernel
def vector_add_naive(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    # Where am I in the grid?
    bx, _, _ = cute.arch.block_idx()    # which block am I in?
    tx, _, _ = cute.arch.thread_idx()   # which thread within the block?
    bd, _, _ = cute.arch.block_dim()    # how many threads per block?
    gd, _, _ = cute.arch.grid_dim()     # how many blocks in the grid?

    # My global index
    idx = bx * bd + tx

    # Stride loop: handles cases where N > total threads
    stride = gd * bd
    i = idx
    while i < N:
        C[i] = A[i] + B[i]
        i += stride


@cute.jit
def solve_naive(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block

    vector_add_naive(A, B, C, N).launch(
        grid=(blocks, 1, 1),
        block=(threads_per_block, 1, 1),
    )
```

### How the Naive Kernel Works

1. **Index calculation**: Each thread computes a unique global index `idx = blockIdx * blockDim + threadIdx`. With 256 threads per block, block 0 handles indices 0–255, block 1 handles 256–511, and so on.

2. **Stride loop**: If the vector has more elements than total threads (which is common for large vectors), each thread processes multiple elements spaced `stride` apart. This pattern is called a **grid-stride loop** — it makes the kernel work for any vector size regardless of grid dimensions.

3. **One element per access**: Each `A[i]` load fetches a single 4-byte float. This is correct but leaves performance on the table.

### Why This Is Slow on H100

The naive kernel achieves roughly 30–40% of peak memory bandwidth on an H100. The reason: each thread issues a separate 4-byte load instruction. The GPU's memory controller must handle 4x more instructions than necessary because we could load 16 bytes (4 floats) in a single instruction.

## Part 2: Optimized for H100

### Understanding the H100 Memory System

The NVIDIA H100 (Hopper architecture) has a memory hierarchy designed for massive throughput:

| Level | Size | Bandwidth | Latency |
|-------|------|-----------|---------|
| HBM3 (Global Memory) | 80 GB | ~3.35 TB/s | ~400 cycles |
| L2 Cache | 50 MB | ~12 TB/s | ~200 cycles |
| L1 / Shared Memory | 256 KB per SM | ~33 TB/s (aggregate) | ~30 cycles |
| Registers | 256 KB per SM | Instant | 1 cycle |

The key bottleneck for vector addition is **HBM3 bandwidth**. The computation (one addition per element) is trivial — the GPU spends almost all its time waiting for data to arrive from memory. This makes vector addition a **memory-bound** kernel.

### Key Insight: Vectorized Memory Access

The H100's memory bus is **5120 bits wide**. When a warp of 32 threads each requests a contiguous 4-byte float, the hardware combines (coalesces) these into a single 128-byte transaction — perfect alignment with a cache line.

But we can do better. Instead of each thread loading one 4-byte float, each thread can load **four floats (16 bytes) in a single instruction**:

```
Naive:      Thread 0 loads A[0]   → 4 bytes, 1 instruction
Vectorized: Thread 0 loads A[0:4] → 16 bytes, 1 instruction (LDG.E.128)
```

This is called a **128-bit vectorized load** (`float4` in CUDA, `LDG.E.128` in PTX assembly). The benefit:
- **4x fewer load instructions** → less instruction scheduler pressure
- **Better bus utilization** → each transaction carries more useful data
- **Same number of memory transactions** → the memory controller was already coalescing, but now the instruction overhead is 4x lower

### What Is Memory Coalescing?

When 32 threads in a warp access addresses that fall within the same 128-byte aligned region, the hardware serves all 32 requests in a **single memory transaction**. This is called coalescing.

```
Good (coalesced):
  Thread 0 → address 0x1000
  Thread 1 → address 0x1004
  Thread 2 → address 0x1008
  ...
  Thread 31 → address 0x107C
  → 1 transaction (128 bytes, fully utilized)

Bad (strided):
  Thread 0 → address 0x1000
  Thread 1 → address 0x2000
  Thread 2 → address 0x3000
  ...
  → 32 separate transactions (128 bytes each, only 4 bytes used per transaction)
```

Vector addition naturally has perfect coalescing because threads access consecutive elements. The optimization is about **reducing instruction count**, not fixing coalescing.

### The Optimized Kernel

```python
@cute.kernel
def vector_add_optimized(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    bx, _, _ = cute.arch.block_idx()
    tx, _, _ = cute.arch.thread_idx()
    bd, _, _ = cute.arch.block_dim()
    gd, _, _ = cute.arch.grid_dim()

    # Each thread processes 4 elements at a time (128-bit vectorized access)
    VECTOR_SIZE = 4
    idx = (bx * bd + tx) * VECTOR_SIZE
    stride = gd * bd * VECTOR_SIZE

    i = idx
    while i + VECTOR_SIZE <= N:
        # Load 4 floats in a single 128-bit transaction
        a_vec = cute.load_vector(A, i, VECTOR_SIZE)  # LDG.E.128
        b_vec = cute.load_vector(B, i, VECTOR_SIZE)  # LDG.E.128

        # Compute
        c_vec = a_vec + b_vec

        # Store 4 floats in a single 128-bit transaction
        cute.store_vector(C, i, c_vec, VECTOR_SIZE)  # STG.E.128
        i += stride

    # Handle remaining elements (tail loop for when N is not divisible by 4)
    tail_idx = bx * bd + tx
    tail_start = (N // VECTOR_SIZE) * VECTOR_SIZE
    i = tail_start + tail_idx
    while i < N:
        C[i] = A[i] + B[i]
        i += gd * bd


@cute.jit
def solve_optimized(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    threads_per_block = 512  # More threads → better latency hiding
    elements_per_thread = 4
    effective_threads_needed = (N + elements_per_thread - 1) // elements_per_thread
    blocks = (effective_threads_needed + threads_per_block - 1) // threads_per_block

    vector_add_optimized(A, B, C, N).launch(
        grid=(blocks, 1, 1),
        block=(threads_per_block, 1, 1),
    )
```

### Why 512 Threads Per Block?

On H100, each SM has 2048 register slots and can run up to 2048 threads. Using 512 threads per block means:
- 4 blocks can run concurrently per SM (4 × 512 = 2048)
- More threads in flight → more memory requests in the pipeline → better **latency hiding**

When one warp is waiting for data from HBM3 (~400 cycles), the SM switches to another ready warp. More warps = more chances to hide memory latency = higher effective bandwidth.

### The Tail Loop

Since we process 4 elements per thread, vectors whose length isn't divisible by 4 have leftover elements. The tail loop handles these stragglers with simple scalar access. For large vectors (millions of elements), the tail loop handles at most 3 elements — negligible overhead.

### CuTe's Abstraction: Tensors and Layouts

In production CuTe code, you wouldn't manually compute indices. CuTe provides **Layout** and **Tensor** abstractions that handle partitioning and vectorization declaratively:

```python
@cute.kernel
def vector_add_cute(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    # Create a 1D tensor view with layout
    layout = cute.make_layout(N)

    # Partition across the grid (each thread gets its slice)
    tA = cute.local_partition(A, layout, threadIdx.x, blockDim.x)
    tB = cute.local_partition(B, layout, threadIdx.x, blockDim.x)
    tC = cute.local_partition(C, layout, threadIdx.x, blockDim.x)

    # Copy with automatic vectorization (CuTe picks LDG.E.128 when alignment allows)
    cute.copy(tA, tC)  # Would need custom op for addition
```

The `cute.copy` primitive automatically selects the widest possible load instruction based on tensor alignment and element type. For `float` tensors aligned to 16 bytes, it emits `LDG.E.128`.

### Performance Expectations

| Kernel | Bandwidth (H100) | % of Peak (3.35 TB/s) |
|--------|-------------------|------------------------|
| Naive (256 threads, scalar) | ~1.2 TB/s | ~36% |
| Vectorized (512 threads, float4) | ~2.9 TB/s | ~87% |
| Theoretical peak | 3.35 TB/s | 100% |

The remaining 13% gap comes from:
- Kernel launch overhead
- Tail loop elements
- L2 cache bank conflicts
- TLB misses on very large vectors

### Summary of Optimizations

| Technique | What It Does | Speedup Factor |
|-----------|-------------|----------------|
| Vectorized loads (128-bit) | 4 floats per instruction instead of 1 | ~1.5–2x |
| More threads (512 vs 256) | Better latency hiding | ~1.2–1.5x |
| Grid-stride with vector width | Correct vectorized iteration | Required for correctness |
| Tail loop | Handle non-aligned remainders | Required for correctness |

## What's Next

Vector addition is memory-bound — the GPU barely computes anything. For compute-bound kernels (matrix multiply, convolution), CuTe's real power shines: **tiled layouts**, **shared memory staging**, and **warp-specialized pipelines**. The patterns you learned here — vectorized access, coalescing, latency hiding — are the foundation for understanding those more complex kernels.
