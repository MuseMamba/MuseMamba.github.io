---
layout: post
title: "Use cute to implement vector addition"
category: 技术
subcategory: CUTLASS
tags: [CUTE, Vector Addition, GPU Programming]
keywords: ["CUTE", "Vector Addition", "GPU Programming", "CUDA", "Performance Optimization"]
description: "Using the CUTE library to implement vector addition on the GPU, demonstrating how to leverage CUTE's features for efficient parallel computation."
reference: "https://leetgpu.com/challenges/vector-addition"
---

# Using CUTE for Vector Addition on the GPU

## Naive Method

- vectorC = vectorA + vectorB.
- N is the length of the vector — the number of valid elements in A, B, and C.
- In the official CuTe DSL docs, @jit is the decorator for host-side JIT-compiled functions, while @kernel is for GPU kernel functions. They are similar to the kernel side api and host side api in the Linux driver development.
- Each thread is a worker. tidx, _, _ = cute.arch.thread_idx() is used to get the unique thread index for each thread.
- A block is a group of threads, one block = one team of workers. bidx, _, _ = cute.arch.block_idx() is used to get the unique block index for each block.
- A grid is the whole set of blocks.
```python
@cute.kernel
def vector_add_kernel(A: cute.Tensor, B: cute.Tensor, C:cute.Tensor, N:cute.Uint32):
    bx, _, _ = cute.arch.block_idx()
    tx, _, _ = cute.arch.thread_idx()
    bd, _, _ = cute.arch.block_dim()
    gd, _, _ = cute.arch.grid_dim()

    idx = bx * bd + tx
    stride = gd*bd

    i = idx
    while i < N:
        C[i] = A[i] + B[i]
        i += stride

# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block

    vector_add_kernel(A,B,C,N).launch(
        grid = (blocks, 1,1),
        block = (threads_per_block, 1,1),
    )
```

## Optimized for H100

- CUDA’s programming guide says global memory accesses are naturally supported in sizes of 1, 2, 4, 8, or 16 bytes, and a single access is compiled into a single global memory instruction only when the data size is one of those sizes and the address is naturally aligned.
- 
