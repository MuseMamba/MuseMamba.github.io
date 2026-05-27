---
layout: post
title: "Vector Addition: From Naive CUDA to H100-Optimized"
category: 技术
subcategory: CUTLASS
tags: [CUDA, CUTE, Vector Addition, GPU Programming]
keywords: ["CUDA", "Vector Addition", "GPU Programming", "float4", "CuTe DSL", "Performance Optimization", "H100"]
description: "A beginner-friendly guide to GPU vector addition — from naive CUDA C++ to H100-optimized kernels with float4 vectorization, grid-stride loops, and a CuTe DSL implementation using zipped_divide and TV layouts."
reference: "https://leetgpu.com/challenges/vector-addition"
---

# Vector Addition: From Naive CUDA to H100-Optimized

Vector addition is the "hello world" of GPU programming: `C[i] = A[i] + B[i]` for every element. It's trivially parallel — every element is independent — which makes it the perfect playground to learn how memory access patterns determine GPU performance.

**Why GPU?** A CPU processes elements one at a time (or a few at a time with SIMD). A GPU has thousands of lightweight cores that can process thousands of elements simultaneously. For vector addition on 25 million elements, a CPU iterates 25 million times; a GPU launches 25 million threads that all run at once.

## Background: How a GPU Executes Work

Before diving into code, let's establish the execution model:

- **Thread**: The smallest unit of execution. Each thread runs the same kernel code but on different data.
- **Block**: A group of threads (typically 128–1024) that can share fast on-chip memory and synchronize with each other.
- **Grid**: The collection of all blocks launched for a kernel. The grid covers the entire problem.
- **Warp**: A hardware scheduling unit of 32 threads within a block. All 32 threads in a warp execute the same instruction simultaneously (SIMT).

When you launch a kernel, the GPU maps your grid of blocks onto its Streaming Multiprocessors (SMs). Each SM runs one or more blocks concurrently.

Here's how these levels relate to each other:

```
Grid (entire problem)
├── Block 0  (up to 1024 threads)
│   ├── Warp 0  (threads 0–31)
│   ├── Warp 1  (threads 32–63)
│   └── ...
├── Block 1
│   ├── Warp 0
│   └── ...
└── ...
```

You specify the grid and block sizes at launch time. The hardware handles warps automatically — you never create warps explicitly.

## Understanding the H100 Memory System

The NVIDIA H100 (Hopper architecture) has a memory hierarchy designed for massive throughput:

| Level | Size | Bandwidth | Latency |
|-------|------|-----------|---------|
| HBM3 (Global Memory) | 80 GB | ~3.35 TB/s | ~400 cycles |
| L2 Cache | 50 MB | ~12 TB/s | ~200 cycles |
| L1 / Shared Memory | 256 KB per SM | ~33 TB/s (aggregate) | ~30 cycles |
| Registers | 256 KB per SM | Instant | 1 cycle |

Vector addition is **memory-bound** — the computation (one add per element) is trivial. The GPU spends almost all its time waiting for data from HBM3. Our job is to maximize memory bandwidth utilization.

**What does "memory-bound" mean?** Every kernel is bottlenecked by either computation (math instructions) or memory (loading/storing data). Vector addition does 1 add per 12 bytes transferred (read A, read B, write C — each 4 bytes for float32). The GPU can do far more math than this in the time it takes to fetch the data, so the memory bus is the bottleneck. Optimizing a memory-bound kernel means getting data in and out of HBM as fast as possible.

## Part 1: The Naive Kernel

A CUDA kernel is a function that runs on the GPU. The `__global__` keyword tells the compiler this function is called from the CPU but executes on the GPU. Every thread runs the same kernel code, but each thread has a different `threadIdx` and `blockIdx` — built-in variables that let each thread figure out which element it should work on.

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

The `solve` function is the **host-side** launcher — it runs on the CPU. The `<<<blocks, threads_per_block>>>` syntax is CUDA's kernel launch syntax: it tells the GPU how many blocks to create and how many threads per block. The formula `(N + threads_per_block - 1) / threads_per_block` is ceiling division — it ensures we launch enough threads to cover all N elements, even when N isn't perfectly divisible by the block size.

### How It Works

1. **Index calculation**: Each thread computes its unique global index `idx = blockIdx.x * blockDim.x + threadIdx.x`. Think of it like this: if you have 256 threads per block, then block 0 has threads 0–255, block 1 has threads 256–511, and so on. `blockIdx.x * blockDim.x` gives the starting index of the block, and `threadIdx.x` is the thread's position within that block.

2. **Bounds check**: `if (idx < N)` prevents out-of-bounds access when N isn't a multiple of the block size. For example, if N = 1000 and we launch 4 blocks of 256 threads (1024 threads total), the last 24 threads have `idx >= 1000` and must do nothing.

3. **Coalesced access**: Adjacent threads access adjacent memory addresses, so warps read contiguous memory — the hardware coalesces these into efficient 128-byte transactions.

### Why This Is Slow

The naive kernel typically achieves only 30–40% of peak HBM3 bandwidth on H100. The problem is **instruction overhead**: each thread issues one 4-byte load instruction per element. The memory controller coalesces warp-level accesses, but the instruction scheduler is overwhelmed by the sheer number of individual load/store instructions.

To understand why, consider what each thread does: it issues 2 load instructions (one for `A[idx]`, one for `B[idx]`), 1 add, and 1 store. That's 3 memory instructions per element, each moving only 4 bytes. The GPU's instruction pipeline has limited throughput — it can only dispatch so many instructions per cycle. When every instruction moves so little data, the pipeline becomes the bottleneck, not the memory bus itself.

## Part 2: Vectorized with `float4`

### Key Insight: Vectorized Memory Access

The GPU memory bus is wide — each memory transaction fetches 128 bytes (a full cache line). In the naive kernel, each thread requests only 4 bytes, but the hardware still fetches the full 128-byte cache line. The data isn't wasted (neighboring threads use adjacent parts of the same cache line), but each thread pays the cost of issuing a separate instruction for just 4 bytes.

**Vectorization** means loading more data per instruction. Instead of each thread loading one 4-byte float, each thread can load **4 floats (16 bytes) in a single 128-bit instruction**:

```
Naive:      Thread 0 loads A[0]     → 4 bytes,  1 instruction
Vectorized: Thread 0 loads A[0:4]   → 16 bytes, 1 instruction (LDG.E.128)
```

Benefits:
- **4x fewer load instructions** → less instruction scheduler pressure
- **Better bus utilization** → each instruction carries more useful data
- **More in-flight bytes** → better latency hiding per Little's Law

### The Vectorized Kernel

The idea is simple: treat the `float` array as an array of `float4` (a struct of 4 floats), so each thread processes 4 elements at once. Since N might not be divisible by 4, we handle the leftover "tail" elements separately with scalar loads.

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

`float4` is a CUDA built-in type — a struct with four members (`x`, `y`, `z`, `w`), each a 32-bit float, 16 bytes total. It's naturally aligned to 16 bytes in memory. When you load a `float4`, the compiler emits a single `LDG.E.128` instruction that fetches all 16 bytes in one shot — the same work that would take 4 separate instructions with scalar `float` loads.

The `reinterpret_cast<const float4*>(A)` tells the compiler to treat the float array as an array of `float4`. This doesn't copy or rearrange any data — it just changes how the pointer arithmetic works. `A4[idx]` now accesses 4 contiguous floats starting at `A[idx * 4]`, and the compiler knows it can use a 128-bit load.

### What Is Memory Coalescing?

GPUs don't fetch individual bytes from memory. Instead, the memory controller works in **transactions** — fixed-size chunks of 128 bytes (a cache line). When 32 threads in a warp access addresses within the same 128-byte aligned region, the hardware serves all requests in a **single memory transaction**:

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

In Parts 1 and 2, we launched one thread per element (or per `float4` chunk). For N = 25 million, that means launching ~6.25 million threads — which means ~24,400 blocks. This works, but it's not ideal: launching that many blocks has overhead, and the GPU only has 132 SMs. Most blocks sit in a queue waiting for an SM to become available.

The **grid-stride loop** pattern takes a different approach: launch a **fixed, modest number** of threads, and have each thread process **multiple elements** by walking through the array in strides.

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

The key line is `for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < vec_N; idx += stride)`. Each thread starts at its global index and jumps forward by `stride` (the total number of threads in the grid) on each iteration. This guarantees every element is covered exactly once, and adjacent threads always access adjacent memory — preserving coalescing.

With N = 25,000,000 and `float4`, there are 6,250,000 vector elements to process. A grid of 256 blocks × 256 threads = 65,536 threads, so each thread handles ~95 `float4` chunks. Here's a visual of how threads walk through the array:

```
Thread 0:  processes elements 0, 65536, 131072, ...
Thread 1:  processes elements 1, 65537, 131073, ...
Thread 2:  processes elements 2, 65538, 131074, ...
...
Thread 65535: processes elements 65535, 131071, ...
```

Benefits:

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

---

# CuTe DSL Version

The same vector addition problem can be expressed using NVIDIA's **CuTe DSL** — a Python-embedded domain-specific language from the [CUTLASS](https://github.com/NVIDIA/cutlass) library. CuTe provides higher-level abstractions (`zipped_divide`, TV layouts, `.load()`) that compile down to the same PTX instructions as the raw CUDA C++ above, but let you manipulate tensor layouts algebraically.

**Why CuTe?** Raw CUDA gives you full control, but as kernels grow more complex (GEMM, attention, convolution), manually managing tile shapes, thread-to-data mappings, and memory access patterns becomes error-prone. CuTe lets you express these patterns declaratively — you describe the layout, and CuTe generates the indexing math for you.

**CuTe uses float16** in the examples below (vs. float32 in the CUDA C++ section above). With float16, each element is 2 bytes, so a 128-bit (16-byte) load fetches 8 elements instead of 4.

## Setup

CuTe DSL programs use PyTorch tensors as input and the `cutlass` Python package for kernel compilation. `from_dlpack` converts a PyTorch tensor into a CuTe tensor without copying data — it just wraps the same GPU memory with CuTe's layout metadata.

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
```

## Part 1: Naive CuTe Kernel

CuTe separates GPU code into two pieces: a **kernel** (`@cute.kernel`) that runs on the GPU, and a **host function** (`@cute.jit`) that runs on the CPU to configure and launch the kernel. This is analogous to the `__global__` kernel + host `solve` function in CUDA C++.

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

CuTe uses a two-step process: `cute.compile` JIT-compiles the kernel to GPU machine code (PTX → SASS), and then you call the compiled function. The `assumed_align=16` parameter is critical — it tells the compiler that the memory pointers are 16-byte aligned, which is required for the compiler to emit vectorized 128-bit load/store instructions later.

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

## Part 2: Vectorized with `zipped_divide`

In the CUDA C++ version, we used `float4` and `reinterpret_cast` to manually group 4 floats into a single 128-bit load. CuTe provides a higher-level abstraction for the same idea: `cute.zipped_divide(tensor, tiler)` partitions a tensor into fixed-size tiles. For vectorization, the tiler specifies how many contiguous elements each thread should access — 8 float16 elements = 16 bytes = one 128-bit load.

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

### The Vectorized Kernel

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

In the previous approaches, the kernel itself computed which data each thread should access (the `mi`, `ni` index math). With TV layout, that mapping is defined **outside the kernel** as a layout object and passed in. The kernel simply says "give me my data" — it doesn't need to know how the data was partitioned.

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

The host constructs a TV layout that maps 256 threads to a `(64, 512)` tile, with each thread loading 16 contiguous bytes per row across multiple rows.

Breaking down the layout construction:
- **Thread layout** `(4, 64)`: arranges 256 threads (4 × 64) into a 2D grid — 4 rows of 64 threads. The `order=(1, 0)` means threads are numbered along columns first (thread 0 and 1 are in the same row, adjacent columns).
- **Value layout** `(16, 16)`: each thread handles a 16×16-byte region. `recast_layout` converts the byte-level layout to element-level (for float16: 16 bytes = 8 elements per column).
- **`make_layout_tv`**: combines thread and value layouts into a single TV layout and returns the tile shape `tiler_mn` that `zipped_divide` needs.

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

### CuTe DSL Performance

| Kernel | Bandwidth (H100, float16) | % of Peak (3.35 TB/s) |
|--------|---------------------------|------------------------|
| Naive (scalar loads) | ~1.0–1.3 TB/s | ~30–40% |
| Vectorized (zipped_divide) | ~2.8–3.1 TB/s | ~85–93% |
| TV Layout (production) | ~2.9–3.1 TB/s | ~87–93% |
| Theoretical peak | 3.35 TB/s | 100% |

### CuTe DSL Summary

| Technique | What It Does | Why It Helps |
|-----------|-------------|--------------|
| `zipped_divide(tensor, (1, 8))` | Tiles tensor so each thread gets 8 elements | Enables 128-bit vectorized load/store |
| `.load()` on tiled slice | Emits single `LDG.E.128` instruction | 8x fewer instructions vs scalar |
| `from_dlpack(t, assumed_align=16)` | Guarantees 16-byte pointer alignment | Required for compiler to emit vectorized instructions |
| `cute.composition(tensor, tv_layout)` | Maps (thread, value) → physical address | Decouples access pattern from kernel logic |

## What's Next

Vector addition is memory-bound — the GPU barely computes anything. For compute-bound kernels (matrix multiply, convolution), the real optimization challenges appear: **tiled layouts**, **shared memory staging**, and **warp-specialized pipelines**. The patterns you learned here — `float4` vectorization in raw CUDA and `zipped_divide` / TV layouts in CuTe — are the exact building blocks used in high-performance GEMM and Flash Attention implementations.
