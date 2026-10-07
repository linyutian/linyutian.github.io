---
title: "Inference Engine vs. Operating System: What Makes LLM Inference Hard?"
date: 2026-10-05T03:59:14-04:00
draft: false
lastmod: 2026-10-06
tags: [llm-inference, gpu, systems]
summary: "Comparing the inference engine to an operating system shows what makes LLM inference a distinct systems problem."
---

A crash course on LLM inference usually starts with the distinction between prefill and decode: prefill is compute-bound, decode is memory-bound. The distinction is real, but it is a symptom rather than a cause. It also leaves out much of what makes inference hard, including the size of the program, the shape of its state, and the fact that large models have to be spread across machines that talk to each other on every token.

The most useful lens I have found for understanding these gaps comes from [Professor Nir Shavit's](https://people.csail.mit.edu/shanir/) lecture on LLM inference, in which he compares and contrasts the inference engine with an operating system. The two are similar in role: both act as a layer that abstracts hardware complexity away from software developers. Where the comparison breaks, it exposes nuances about the inference workload.

This post uses that comparison and contrast to understand the "inference problem": what its constraints are, why they make it challenging, and what it means for product and business.

## 1. The Inference Engine as an Operating System

An operating system is an abstraction layer. A hardware vendor writes a driver without knowing which database will run on it, and the database team writes code without knowing which CPU sits underneath. That separation is valuable enough to be a business in its own right, which is how companies like Red Hat make money.

An inference engine such as vLLM, SGLang or TensorRT-LLM occupies a similar position in the AI stack. It sits between models and accelerators, so a model author does not need to write kernels for every new chip, and a chip vendor does not need to hand-tune every new model.

![Two side-by-side stacks. Left: applications such as databases, on an OS and runtime, on CPUs. Right: AI applications such as LLMs, on an inference engine, on GPUs, TPUs and other accelerators. The middle layer in each hides hardware from applications and applications from hardware.](os-vs-inference-stack.svg)

The inference engine also relies on the same four mechanisms as an OS:

| Mechanism | The OS manages | The inference engine manages |
|---|---|---|
| Scheduling | processes, threads | requests, tokens |
| Virtual memory | pages in RAM | KV-cache blocks in HBM |
| Resource allocation | CPU, memory | GPU memory, compute, network |
| Caching | page cache, buffer cache | KV cache, prefix cache |

Many of the algorithms carry over too. For example, vLLM's PagedAttention is virtual-memory paging applied to the KV cache ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)). What's different between the OS and the inference engine is the unit these mechanisms manage.

That unit is the **token**. An OS manages processes: long-lived jobs whose demands on CPU and memory stay roughly stable. An inference engine manages requests, but measures work in tokens. At every step, the scheduler decides how many tokens each request processes. The KV cache grows by a fixed amount per token and is allocated in blocks of tokens. Prefix caches are keyed on token sequences, and prices and latency are quoted per token. Unlike a process's demands, a request's demands change with every token it generates.

Because the workloads differ, the OS toolbox cannot be reused as is. Each OS algorithm was designed around assumptions about the OS workload. Scheduling assumes switching between jobs is cheap. Paging assumes a process touches only a small working set at a time. Caching assumes recently used data will be used again and fits close to the processor. Whether a tool still works for inference depends on whether its assumptions still hold, so we first need to understand the inference workload.

## 2. Three Differences from the OS Workload

Compared with LLM inference, operating systems and the workloads they handle have not changed much. CPUs got faster and wider; databases kept the same tables, queries and transactions. So the assumptions behind OS mechanisms still hold for most software: the program is small, the data streams through it, and the state is shared. LLM inference differs on all three.

### 2.1 Huge Program, Small Input

In classic software, a small program runs over large data. If we treat a model's weights as the program and the prompt as its input, the ratio flips by many orders of magnitude:

| Use case | Workload | Program ÷ input |
|---|---|---|
| A bank totaling monthly statements | PostgreSQL (~10 MB) scanning a 100 GB table | ~10⁻⁴× |
| An app counting DAU from a year of logs | Spark job (a few hundred MB) over a 1 TB log archive | ~10⁻⁴–10⁻³× |
| A store writing alt text for a product photo | Qwen2.5-VL-7B (~16 GB) captioning a ~1 MB image | ~16,000× |
| A student solving one middle-school math problem | Llama 3.1 70B (~140 GB) answering a ~100-token prompt | ~350 million× |

*Program size in parentheses; model weights in BF16; text at about 4 bytes per token.*

The bottom two rows make the point. In inference workloads, the program is thousands to hundreds of millions of times larger than its input, the reverse of the classic case. This matters for the economics of scaling LLM usage. In a classic *input-driven* program, cost is a function of input size, and that is what systems engineers optimize around. In a *parameter-driven* program like an LLM, cost is dominated by the size of the program itself. A model of this size cannot stay resident in any cache, and the largest models do not fit on a single accelerator, so they have to be sharded.

![Program size relative to input. Classic system: a small program of about 10 MB runs over 100 GB to 1 TB of input data. LLM inference: the program, the model weights, is about 16 to 140 GB, while the prompt is kilobytes to a megabyte, so the program is thousands to hundreds of millions of times larger than its input.](program-vs-input.svg)

### 2.2 The Program Moves, Not the Data

"Program" and "data" need precise meanings here. In classic data processing, the program is the code and the data is the input records. The code is small and stays close to the processor, in its instruction cache. The records stream from disk or memory through the code, and each is touched once or a few times. When a database scans a table, almost every byte that moves is data.

In LLM inference, the program is the model weights and the data is the current token's hidden state: one 1 × 8,192 vector, about 16 KB per pass. The model weights are far too large to stay on chip, so every pass reads all of them again from HBM. To move 16 KB of data forward by one token for a 70B model, the GPU reads 140 GB of program: about ten million bytes of program for every byte of data. The data barely moves; the program streams through the processor, once per token.

![What moves through the processor. Classic system: data records stream through a small program that stays resident. LLM inference: about 140 GB of weights stream from HBM into compute on every token, while the data is a single token vector of about 16 KB.](program-moves.svg)

### 2.3 Private, Growing State

A database's state is large, persistent and **shared**; every query reads from the same tables. An LLM request's state is its KV cache. It is **private** to one request, it **grows** with every token, and every decode step **reads all of it and appends to it**.

![How state is held. Classic system: many queries read one large, persistent set of shared tables. LLM inference: each request has its own private KV cache, shown as bars of different lengths for requests A, B and C, each growing with every token.](private-state.svg)

The engine therefore manages two memory objects with very different properties:

| | Model weights | KV cache |
|---|---|---|
| Size | huge, fixed, known in advance | grows per token; final size unknown |
| Sharing | shared by every request | private to one request |
| Lifetime | as long as the model is loaded | as long as the request (or longer, if cached) |

The three differences build on each other. The program is huge (2.1), so it cannot stay near the processor. Decoding therefore streams it through the processor on every token (2.2). Each request then adds its own growing state (2.3), which competes for the same memory. Together they make memory, not compute, the scarce resource.

## 3. Two Consequences of the Differences

### 3.1 Locality of Reference, Relocated

Locality of reference is the principle that programs tend to reuse data they used recently (temporal locality) and data stored near it (spatial locality). Caches exist because of it, and so does the distributed-systems rule to move computation to the data. In LLM inference, locality loses much of its traditional role, but it does not disappear. It moves, and each difference from Section 2 moves it somewhere specific.

**A. Across machines, locality fails** (2.1). A database query touches only the shard that holds its data, but every token passes through every shard of a model too large for one accelerator. Since computation cannot move to the data, the main lever is faster communication, which is why interconnects such as NVLink and rack-scale systems such as NVL72 are now part of the inference runtime.

**B. Inside the chip, locality matters more** (2.2). Every token reuses all of the weights, but 140 GB cannot fit in about 50 MB of L2 cache, so each pass streams them from HBM. Kernels therefore recover locality one level down: FlashAttention tiles attention so its working set stays in on-chip SRAM ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)), and kernel fusion keeps intermediate results on chip between operations.

**C. Across requests, locality returns** (2.3). A request's KV cache is private, yet many requests share a prefix: a system prompt, the earlier turns of a conversation, or the context an agent resubmits at every step. Prefix caching, such as SGLang's RadixAttention ([Zheng et al., 2024](https://arxiv.org/abs/2312.07104)), keeps the KV cache of shared prefixes to avoid recomputation, which is extremely valuable in agentic workloads.

### 3.2 Idle Compute, Scarce Memory

In the introduction, I called the prefill/decode distinction a symptom. With the three differences in hand, we can trace it to its cause.

Today's mainstream hardware is not purpose-built for LLM inference: the GPU was shaped by graphics and ML training, workloads that perform many operations on every byte they fetch from memory. Whether inference keeps that compute busy depends on how much data each pass of the program serves (2.2):

- **Prefill** processes the whole prompt in one pass, so each weight read serves every prompt token. It is **compute-bound**.
- **Decode** generates one token per pass, so each weight read serves a single vector and the compute waits on memory. It is **memory-bandwidth-bound**.

The two phases are neither the same workload nor entirely different: both stream the same huge program (2.1) through the processor (2.2), but for a whole prompt in one phase and a single token in the other.

A single request cannot saturate GPU compute in a decode pass, because each token depends on the one before it (speculative decoding is the exception). So the inference engine batches requests: since they all run the same weights, one weight read serves many of them. Yet the private, growing KV cache (2.3) keeps the batch smaller than it ideally would be:

- **The cache is not shared.** Each request's attention reads its own KV cache, so that part of decode stays memory-bound at any batch size.
- **The cache competes for memory.** Every request's cache grows in the same HBM as the weights, so memory runs out before the batch is large enough to saturate compute.

Decode thus leaves compute idle while memory runs short; that, more than the prefill/decode split itself, is the core inefficiency. Much of a modern inference engine works around it, with each technique aimed at one of the three differences from Section 2:

| Technique | Difference | What it works around |
|---|---|---|
| Weight quantization (FP8, INT4) | Huge program (2.1) | **Bytes per weight:** fewer bits to read per pass. |
| Continuous batching ([Yu et al., 2022](https://www.usenix.org/conference/osdi22/presentation/yu)) | Program moves (2.2) | **Half-empty batches:** requests join and leave at every step. |
| Speculative decoding ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)) | Program moves (2.2) | **One token per pass:** several drafted tokens verified at once. |
| Prefill/decode disaggregation ([Zhong et al., 2024](https://arxiv.org/abs/2401.09670)) | Program moves (2.2) | **Opposite bottlenecks:** each phase on hardware suited to it. |
| Paged KV cache ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)) | Growing state (2.3) | **Fragmented KV memory:** cache allocated in small blocks on demand. |
| Grouped-query attention ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) | Growing state (2.3) | **Bytes per cached token:** heads share keys and values. |

## 4. Defining the Inference Problem

Taken together, the three differences give a precise definition of the inference problem:

> **LLM inference is the problem of streaming a huge program through limited memory, once per token, for many requests whose private state keeps growing.**

Each part of that sentence is a constraint:

- **A huge program.** The weights are too large to stay near the compute, and the largest models must be split across machines that communicate on every token.
- **Streamed once per token.** Every output token rereads the weights, so memory bandwidth, not arithmetic, sets the cost per token.
- **Many requests.** Sharing one weight read across requests is the main way to recover efficiency.
- **Private, growing state.** Each request's KV cache grows to a size nobody knows in advance and competes with the weights for the same memory, which caps how many requests can share one weight read.

This closes the loop on Section 1, where each OS mechanism rested on an assumption about its workload. The table follows each assumption through the three differences.

| Mechanism | OS assumption | What happens in LLM inference | Problem to solve |
|---|---|---|---|
| Scheduling | Switching jobs is cheap; more jobs means more contention. | Preemption evicts gigabytes of KV cache; more requests make each token cheaper. | How to trade throughput against latency when output lengths are unknown? |
| Virtual memory | A process touches a small working set; paging adds isolation and overcommit. | Every decode step reads the entire KV cache; paging exists because output length is unknown. | How to allocate, share and reclaim KV memory when it sets batch size? |
| Resource allocation | Computation moves to the data. | The program is distributed and communicates on every token; prefill and decode want different hardware. | How to split a model across devices and place the two phases? |
| Caching | Recent data is reused and fits close to the processor. | Weights are reused but do not fit; locality moves into kernels and across requests. | Where should KV state live, for how long, and how should requests be routed to it? |

One more difference makes the inference engine's job harder than the OS's. An OS sits between a stable floor and a stable ceiling; an inference engine has neither. Accelerators (GPUs, TPUs) change every one to two years, the dominant workload changes even faster (from chat to tool use, reasoning, agents and video), and new models arrive almost every quarter. Today's inference engine is therefore an **adaptation layer**, not just an abstraction layer.

## 5. Product and Business Implications

**Scale wins on cost.** Because one weight read serves a whole batch, cost per token falls with sustained concurrency. Operators with large, steady traffic keep batches full; small or bursty operators pay for idle bandwidth on the same hardware. This favors large model providers (e.g., OpenAI, Anthropic and Google) and shared serving platforms (e.g., Together AI and Fireworks AI) over dedicated per-application deployments.

**Hardware is splitting by phase.** Prefill wants compute; decode wants memory bandwidth. Disaggregated serving ([llm-d](https://github.com/llm-d/llm-d), [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)), SRAM-heavy decode chips (Groq, Cerebras) and NVIDIA's prefill-oriented Rubin CPX, which uses GDDR7 instead of HBM ([The Register, 2025](https://www.theregister.com/2025/09/10/nvidia_rubin_cpx/)), all point the same way: the right mix of chips for a workload matters more than a single "best inference chip".

**State is becoming a product.** Prefix caching already shows up in pricing: several API providers charge much less for cached input tokens than for uncached ones. As agents make context reuse the norm, managing KV state (where it lives, how long it is kept, how requests are routed to it) becomes a competitive capability, and systems such as [LMCache](https://github.com/LMCache/LMCache) treat it as data to store and move.

## 6. Open Questions

The third difference looks the least permanent. KV state used to be private and short-lived, but prefix caching, long conversations and agent loops are making it longer-lived, shared and stored across machines, much like database state. That leaves two questions I am still thinking about:

- **Is the inference engine becoming more like a database than an OS?** If managing persistent, shared context becomes the dominant cost, storage and databases may be the more relevant design tradition.
- **Who owns the state?** If a long-running agent's KV cache is the most valuable thing in the system, it matters whether it lives with the model provider, the serving platform or the application.

## References

- Ainslie, J., Lee-Thorp, J., de Jong, M., et al. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). EMNLP 2023.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). NeurIPS 2022.
- Kwon, W., Li, Z., Zhuang, S., et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP 2023.
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). ICML 2023.
- Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., & Chun, B.-G. (2022). [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu). OSDI 2022.
- Zheng, L., Yin, L., Xie, Z., et al. (2024). [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). NeurIPS 2024.
- Zhong, Y., Liu, S., Chen, J., et al. (2024). [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670). OSDI 2024.
- Rubin CPX: [The Register](https://www.theregister.com/2025/09/10/nvidia_rubin_cpx/) (Sept. 10, 2025); [Futurum Group](https://futurumgroup.com/insights/nvidias-new-rubin-cpx-targets-future-of-large-scale-inference/).
