---
layout: home
title: Ren Zezhong · AI Infrastructure & Systems Security
description: PhD student at UCAS working on LLM inference, agent runtime systems, and kernel security. Research, selected engineering work, and technical writing.
mathjax: false
---

<section id="about" class="hero">
  <p class="eyebrow">AI Infrastructure · Agent Systems · Systems Security</p>
  <h1>Ren Zezhong <span lang="zh-CN">任泽众</span></h1>
  <p class="hero-intro">I study and build systems for efficient, reliable, and secure AI.</p>
  <div class="bio">
    <p>I am a PhD student in Computer Science at the <strong>University of Chinese Academy of Sciences (UCAS)</strong>, where I began my doctoral studies in September 2022. From March 2024 to March 2025, I was a joint-training PhD student at <strong>EPFL</strong>, working with Mathias Payer.</p>
    <p>My work spans LLM inference runtimes, agent infrastructure, and systems security—from monitoring and isolating agent execution to exploring under-tested paths in the Linux kernel.</p>
  </div>
  <div class="contact-links" aria-label="Contact">
    <a class="primary-link" href="mailto:{{ site.email }}">{{ site.email }}</a>
  </div>
  <p class="availability">Seeking AI infrastructure and agent systems opportunities for the 2027 graduate hiring cycle, including internships with a path to full-time roles.</p>
</section>

<section id="work" class="home-section">
  <div class="section-heading"><h2>Selected Work</h2><span>Research & engineering</span></div>
  <div class="work-grid">
    <article class="work-item">
      <span class="work-number" aria-hidden="true">01</span>
      <h3>LLM inference efficiency & reliability</h3>
      <p>Investigating small-model supervised fine-tuning for input preprocessing and its effect on downstream inference efficiency and stability. Studying non-crash failures—including repetition, empty outputs, and semantic degradation—when inference optimizations interact with particular prompt patterns.</p>
      <p class="work-topics">Small-model SFT · Inference runtimes · Robustness</p>
    </article>
    <article class="work-item">
      <span class="work-number" aria-hidden="true">02</span>
      <h3>Agent evaluation infrastructure</h3>
      <p>Building agents for automated vulnerability discovery and validation, alongside parallel evaluation infrastructure tested with CyberGym. The framework combines dynamic scheduling, retries for infrastructure failures, and per-case isolation to keep timeouts, OOMs, and process failures from blocking an evaluation run.</p>
      <p class="work-topics">Dynamic scheduling · Failure recovery · Isolation</p>
    </article>
    <article class="work-item">
      <span class="work-number" aria-hidden="true">03</span>
      <h3>Agent runtime monitoring & security</h3>
      <p>Developing an Agent Runtime Monitor that connects agent traces with OS-level process, file, network, and resource activity. Configurable policies control resource quotas and access permissions; collected traces and feedback provide a foundation for future agent training and process rewards.</p>
      <p class="work-topics">Runtime observability · Access control · Training feedback</p>
    </article>
  </div>
</section>

<section id="publications" class="home-section">
  <div class="section-heading"><h2>Publications</h2><span>Fuzzing & systems security</span></div>
  <article class="publication">
    <p class="publication-venue">NDSS 2026 <span>First author</span></p>
    <h3><a href="https://www.ndss-symposium.org/wp-content/uploads/2026-s921-paper.pdf">SYSYPHUZZ: the Pressure of More Coverage</a></h3>
    <p class="authors"><strong>Zezhong Ren</strong>, Han Zheng, Zhiyao Feng, Qinying Wang, Marcel Busch, Yuqing Zhang, Chao Zhang, Mathias Payer</p>
    <p>Kernel fuzzing that prioritizes under-explored code regions through selective task scheduling and context-preserving mutation.</p>
    <div class="publication-links"><a href="https://www.ndss-symposium.org/wp-content/uploads/2026-s921-paper.pdf">Paper ↗</a><a href="https://github.com/HexHive/Sysyphuzz">Code ↗</a></div>
  </article>
  <article class="publication">
    <p class="publication-venue">USENIX Security 2023 <span>Fourth author</span></p>
    <h3><a href="https://www.usenix.org/conference/usenixsecurity23/presentation/zheng">FISHFUZZ: Catch Deeper Bugs by Throwing Larger Nets</a></h3>
    <p class="authors">Han Zheng, Jiayuan Zhang, Yuhang Huang, <strong>Zezhong Ren</strong>, He Wang, Chunjie Cao, Yuqing Zhang, Flavio Toffalini, Mathias Payer</p>
    <p>Distance-aware input prioritization and dynamic target ranking to explore deeper program behavior.</p>
    <div class="publication-links"><a href="https://www.usenix.org/conference/usenixsecurity23/presentation/zheng">Paper & talk ↗</a><a href="https://github.com/HexHive/FishFuzz">Code ↗</a></div>
  </article>
  <article class="publication">
    <p class="publication-venue">Journal of Computer Research and Development · 2021 <span>First author</span></p>
    <h3><a href="https://crad.ict.ac.cn/article/doi/10.7544/issn1000-1239.2021.20201018">A Review of Fuzzing Techniques</a></h3>
    <p class="authors" lang="zh-CN"><strong>任泽众</strong>，郑晗，张嘉元，王文杰，冯涛，王鹤，张玉清</p>
    <p>A survey of fuzzing workflows and techniques, with attention to kernel and IoT security.</p>
    <div class="publication-links"><a href="https://crad.ict.ac.cn/article/doi/10.7544/issn1000-1239.2021.20201018">Paper (中文) ↗</a></div>
  </article>
</section>

<section id="writing" class="home-section">
  <div class="section-heading"><h2>Selected Writing</h2><a href="{{ '/pages/writing.html' | relative_url }}">All posts <span aria-hidden="true">→</span></a></div>
  <p class="section-intro">Notes from working through systems, from first principles to implementation.</p>
  <div class="writing-list">
    <a class="writing-item" href="{% post_url /tech/AIInfra/InferenceEngine/Sglang/2026-03-11-sglang-mm %}"><span class="writing-topic">LLM systems</span><h3>SGLang Memory Management & KV Cache</h3><p>Memory pools, token-to-KV mapping, and the mechanics behind inference serving.</p><span class="writing-date">March 2026 <span aria-hidden="true">↗</span></span></a>
    <a class="writing-item" href="{% post_url /tech/CUTLASS/2026-04-21-cute-vectoradd %}"><span class="writing-topic">GPU programming</span><h3>Vector Addition: From Naive CUDA to H100-Optimized</h3><p>Vectorization, memory access patterns, and a CuTe DSL implementation.</p><span class="writing-date">April 2026 <span aria-hidden="true">↗</span></span></a>
    <a class="writing-item" href="{% post_url /tech/Linux/2026-05-27-linux-namespaces %}"><span class="writing-topic">Linux internals</span><h3>Linux Namespaces: From First Principles to Kernel Implementation</h3><p>How Linux partitions resources, from userspace APIs to kernel data structures.</p><span class="writing-date">May 2026 <span aria-hidden="true">↗</span></span></a>
  </div>
</section>

<aside class="personal-note">
  <p>PhD = Prolonged Hard Days. <em>But Progress Happens Daily.</em></p>
  <a href="{{ '/pages/about.html' | relative_url }}">A little about this blog →</a>
</aside>
