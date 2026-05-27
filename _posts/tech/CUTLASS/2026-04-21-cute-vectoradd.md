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

## Setup

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
```

## Part 1: The Naive Kernel

```python
@cute.kernel
def naive_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n

    a_val = gA[mi, ni]
    b_val = gB[mi, ni]
    gC[mi, ni] = a_val + b_val
```

### Host Launch Function

The `@cute.jit` decorator marks a host-side function that configures and launches the kernel. It runs on the CPU and sets up the grid/block dimensions.

```python
@cute.jit
def naive_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    num_threads_per_block = 256
    m, n = mA.shape

    naive_elementwise_add_kernel(mA, mB, mC).launch(
        grid=((m * n) // num_threads_per_block, 1, 1),
        block=(num_threads_per_block, 1, 1),
    )
```

### Running It

```python
M, N = 16384, 8192

a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

naive_fn = cute.compile(naive_elementwise_add, a_, b_, c_)
naive_fn(a_, b_, c_)

torch.testing.assert_close(c, a + b)
```

### How It Works

1. **Index calculation**: Each thread gets a unique global index `thread_idx = bidx * bdim + tidx`, then maps it to 2D coordinates `(mi, ni)` into the tensor.

2. **1:1 mapping**: Each thread processes exactly one element. This is simple but means each load is a single scalar (2 bytes for float16).

3. **Coalesced access**: Adjacent threads access adjacent columns (`ni = thread_idx % n`), so warps read contiguous memory — the hardware coalesces these into efficient 128-byte transactions.

### Why This Is Slow

The naive kernel typically achieves only 30–40% of peak HBM3 bandwidth on H100. The problem is **instruction overhead**: each thread issues one tiny load instruction per element. The memory controller was already coalescing warp-level accesses, but the instruction scheduler is overwhelmed by the sheer number of individual load/store instructions.

## Part 2: Optimized for H100

### Understanding the H100 Memory System

The NVIDIA H100 (Hopper architecture) has a memory hierarchy designed for massive throughput:

| Level | Size | Bandwidth | Latency |
|-------|------|-----------|---------|
| HBM3 (Global Memory) | 80 GB | ~3.35 TB/s | ~400 cycles |
| L2 Cache | 50 MB | ~12 TB/s | ~200 cycles |
| L1 / Shared Memory | 256 KB per SM | ~33 TB/s (aggregate) | ~30 cycles |
| Registers | 256 KB per SM | Instant | 1 cycle |

Vector addition is **memory-bound** — the computation (one add per element) is trivial. The GPU spends almost all its time waiting for data from HBM3. Our job is to maximize memory bandwidth utilization.

### Key Insight: Vectorized Memory Access

Instead of each thread loading one 2-byte float16, each thread can load **8 float16 values (16 bytes) in a single 128-bit instruction**:

```
Naive:      Thread 0 loads A[0]     → 2 bytes, 1 instruction
Vectorized: Thread 0 loads A[0:8]   → 16 bytes, 1 instruction (LDG.E.128)
```

Benefits:
- **8x fewer load instructions** (for float16) → less instruction scheduler pressure
- **Better bus utilization** → each instruction carries more useful data
- **More in-flight bytes** → better latency hiding per Little's Law

### What Is Memory Coalescing?

When 32 threads in a warp access addresses within the same 128-byte aligned region, the hardware serves all requests in a **single memory transaction**:

```
Good (coalesced):
  Thread 0 → address 0x1000
  Thread 1 → address 0x1002    (float16 = 2 bytes apart)
  Thread 2 → address 0x1004
  ...
  → 1 transaction (128 bytes, fully utilized)

Bad (strided):
  Thread 0 → address 0x1000
  Thread 1 → address 0x2000
  → 32 separate transactions (mostly wasted bytes)
```

Vector addition already has perfect coalescing. Vectorization reduces **instruction count**, not memory transactions.

### The Vectorized Kernel with `zipped_divide`

CuTe provides `cute.zipped_divide(tensor, tiler)` to partition a tensor into fixed-size tiles. For vectorization, the tiler specifies how many contiguous elements each thread should access — 8 float16 elements = 16 bytes = one 128-bit load.

```python
@cute.kernel
def vectorized_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # gA has been tiled by (1, 8) on the host side via zipped_divide.
    # gA.shape is now ((1, 8), (M, N/8)).
    # The 1st mode (1,8) is the per-thread tile (8 contiguous elements).
    # The 2nd mode (M, N/8) indexes which tile each thread works on.
    m, n = gA.shape[1]   # thread-domain shape
    ni = thread_idx % n
    mi = thread_idx // n

    # .load() emits a single 128-bit LDG.E.128 instruction
    a_vec = gA[(None, (mi, ni))].load()
    b_vec = gB[(None, (mi, ni))].load()

    # Elementwise add on the vector, then store (128-bit STG.E.128)
    gC[(None, (mi, ni))] = a_vec + b_vec
```

### Host-Side: Tiling with `zipped_divide`

The host function tiles each tensor before passing them to the kernel. `cute.zipped_divide(mA, (1, 8))` reshapes an `(M, N)` tensor into `((1, 8), (M, N/8))` — grouping every 8 contiguous column elements into a tile.

```python
@cute.jit
def vectorized_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    threads_per_block = 256

    # Tile: each thread handles 8 contiguous float16 = 16 bytes = 128-bit
    gA = cute.zipped_divide(mA, (1, 8))
    gB = cute.zipped_divide(mB, (1, 8))
    gC = cute.zipped_divide(mC, (1, 8))

    vectorized_elementwise_add_kernel(gA, gB, gC).launch(
        grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )
```

### Running the Vectorized Kernel

```python
a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

# assumed_align=16 tells the compiler pointers are 16-byte aligned,
# enabling 128-bit vectorized load/store instructions.
a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

vec_fn = cute.compile(vectorized_elementwise_add, a_, b_, c_)
vec_fn(a_, b_, c_)

torch.testing.assert_close(c, a + b)
```

### How `zipped_divide` Enables Vectorization

Let's trace what happens step by step:

```
Original tensor mA:  shape (M, N), layout (N, 1)
                     e.g. (2048, 2048):(2048, 1)

After zipped_divide(mA, (1, 8)):
  shape:  ((1, 8), (2048, 256))
  layout: ((0, 1), (2048, 8))
           ~~~~~~  ~~~~~~~~~~
             |         |
             |         └── indexes which tile (which thread works on it)
             └── per-thread tile: 8 contiguous elements
```

When thread `t` accesses `gA[(None, (mi, ni))].load()`:
- `(mi, ni)` selects which tile
- `None` means "give me the entire tile" — all 8 elements
- `.load()` compiles to a single `LDG.E.128` instruction because the 8 float16 values are contiguous and 16-byte aligned

## Part 3: Advanced — TV Layout

For production kernels, CuTe offers **TV (Thread-Value) Layout** — a rank-2 layout that maps `(thread_index, value_index)` directly to tensor coordinates. This decouples the thread-to-data mapping from the kernel logic, making it easy to experiment with different access patterns.

```python
@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Slice to get this thread-block's tile
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    # Compose block-local tensor with TV layout:
    # blkA maps (TileM, TileN) → physical address
    # tv_layout maps (tid, vid) → (TileM, TileN)
    # composition maps (tid, vid) → physical address
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    # Slice to get this thread's fragment
    thrA = tidfrgA[(tidx, None)]
    thrB = tidfrgB[(tidx, None)]
    thrC = tidfrgC[(tidx, None)]

    # Vectorized load + add + store
    thrC[None] = thrA.load() + thrB.load()
```

### TV Layout Host Code

The host constructs a TV layout that maps 256 threads to a `(64, 512)` tile, with each thread loading 16 contiguous bytes per row across multiple rows:

```python
@cute.jit
def elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    coalesced_ldst_bytes = 16  # 128-bit = 16 bytes

    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type

    # Thread layout: 4 rows × 64 columns (64 threads read contiguous columns)
    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    # Value layout: each thread reads 16 × 16 bytes, recast to element type
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )
```

### Why TV Layout Matters

The TV layout separates **what each thread does** from **how data is arranged in memory**. You can change the access pattern (e.g., different tile shapes, different thread-to-data mappings) by modifying only the layout construction — the kernel code stays the same. This is the foundation CuTe uses to build high-performance GEMM and attention kernels.

## Performance Expectations

| Kernel | Bandwidth (H100, float16) | % of Peak (3.35 TB/s) |
|--------|---------------------------|------------------------|
| Naive (scalar loads) | ~1.0–1.3 TB/s | ~30–40% |
| Vectorized (zipped_divide) | ~2.8–3.1 TB/s | ~85–93% |
| TV Layout (production) | ~2.9–3.1 TB/s | ~87–93% |
| Theoretical peak | 3.35 TB/s | 100% |

The remaining gap comes from kernel launch overhead, TLB misses on large tensors, and L2 bank conflicts.

## Summary

| Technique | What It Does | Why It Helps |
|-----------|-------------|--------------|
| `zipped_divide(tensor, (1, 8))` | Tiles tensor so each thread gets 8 elements | Enables 128-bit vectorized load/store |
| `.load()` on tiled slice | Emits single `LDG.E.128` instruction | 8x fewer instructions vs scalar |
| `from_dlpack(t, assumed_align=16)` | Guarantees 16-byte pointer alignment | Required for compiler to emit vectorized instructions |
| `cute.composition(tensor, tv_layout)` | Maps (thread, value) → physical address | Decouples access pattern from kernel logic |
| 256 threads per block | Good occupancy, multiple blocks per SM | Hides memory latency via warp switching |

## What's Next

Vector addition is memory-bound — the GPU barely computes anything. For compute-bound kernels (matrix multiply, convolution), CuTe's real power shines: **tiled layouts**, **shared memory staging**, and **warp-specialized pipelines**. The patterns you learned here — `zipped_divide`, TV layouts, vectorized `.load()` — are the exact building blocks used in CuTe's GEMM and Flash Attention implementations.
