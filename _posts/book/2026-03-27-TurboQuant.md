---
layout: post
title: "TurboQuant: Near-Optimal Online Vector Quantization"
tags: [Quantization]
category: 阅读
keywords: [阅读, Google, TurboQuant]
description: "本文介绍 Google 提出的 TurboQuant：一种近似最优的在线向量量化算法，用于在推理阶段压缩 KV cache，适配大规模模型部署。"
---

# TurboQuant: Near-Optimal Online Vector Quantization

- [Paper (Google, arXiv)](https://arxiv.org/pdf/2504.19874)
- [SGLang PR (@shvmjndl)](https://github.com/sgl-project/sglang/pull/21419)

## TL;DR

TurboQuant is a near-optimal **online vector quantization** method for compressing **KV cache** during inference.

It applies a **random rotation** to the original vector (QR-based in the paper; replaced by **FWHT** in the SGLang PR), which makes the rotated vector’s coordinates follow a **Beta distribution** on the sphere. Then it **quantizes each coordinate independently** (MSE objective + QJL term), stores the quantized indices in the KV cache, and reconstructs the vector by applying the inverse rotation during retrieval.

### Key points

- **Random rotation** reduces sensitivity to the original data distribution, making the rotated vectors more uniformly distributed on the sphere; each coordinate follows a **Beta distribution**.
- As the dimension **d** increases, the coordinate distribution concentrates toward **N(0, 1/d)**, and coordinates become approximately independent (in a probabilistic sense).
- With near-independent coordinates, TurboQuant enables **per-coordinate scalar quantization** (e.g., **Lloyd–Max**) to compute near-optimal centroids.
- **QJL** is introduced to mitigate inner-product errors beyond MSE (note: the current PR has not fully integrated this part yet).
- TurboQuant performs better when the model’s **K-norm** is close to **1**, because it assumes the input vectors lie on (or near) the **unit hypersphere**, i.e., \( \|x\|_2^2 = 1 \).  
  This helps explain why **Mistral-7B** (≈ **1.3×**) performs better than **Qwen3** (≈ **2.1×–2.4×**) under TurboQuant.

> Note: This is my personal interpretation and summary. I recommend reading the original paper and the PR side by side.  
> In my view, AI infrastructure is one of the most tightly coupled areas between academia and industry: mathematical elegance is like an artist’s ideas, while AI infra is the craft—brushes, canvas, and technique. Its impact will likely keep expanding, much like the ripple effects of the Industrial Revolution.
