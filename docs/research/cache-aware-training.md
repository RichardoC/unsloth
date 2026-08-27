# Cache-Aware Training: Research Report

Working name: `cache-aware-training`. Date: 2026-08-27. Status: research only, no product code.

Companion: [`cache-aware-weight-layout.md`](cache-aware-weight-layout.md) investigates the narrower framing -- reordering weights so related parts sit together, output-preserving by construction. Both reports land on the same first deliverable (the fidelity harness of section 3 below).

## Summary

The idea — buy inference-time KV-cache efficiency with a cheap, output-preserving finetune — is real and well-precedented (GQA uptraining, DMC, SwiftKV, TransMLA/MHA2MLA, SSM distillation), but the space of techniques is filtered hard by one constraint: **the converted model must still load in llama.cpp as an existing GGUF architecture**, because Unsloth Studio's serving path is llama-server on pinned prebuilt binaries. Only one surveyed transform passes that filter with zero upstream work today: **KV-head narrowing of an existing GQA model (e.g. 8→2 KV heads) with logit-distillation uptraining**, since `n_kv_heads` is plain llama-arch metadata. Everything else (MLA conversion, cross-layer KV sharing, SwiftKV, DMC, SWA-ification of a full-attention model, custom SSM hybrids) requires convert-script and/or llama.cpp C++ work upstream. A second finding tempers expectations: on the Studio target (single user, 1–2 slots, Apple Silicon), the KV cache is *not* the binding constraint below ~16k context — weights memory/bandwidth is — and Studio already exposes quantized KV (`--cache-type-*`, ~2× free) and disables within-sequence eviction (`--no-context-shift`). The measurable win is therefore **memory headroom and decode speed at long context** (8B @ 32k: KV 4 GiB f16 → 1 GiB after 8→2 narrowing; → ~0.5 GiB combined with q8_0), plus the reusable asset this work would create either way: **a teacher-consistency distillation trainer and fidelity eval harness that unsloth does not have today**. Recommendation: ship the harness + GQA-narrowing recipe as the primary slice; treat MLA conversion as the fallback contingent on upstream GGUF support; and say plainly that for users who just want cache-cheap local inference, picking a natively cache-cheap model (Gemma 3's 5:1 SWA interleave, Qwen3-Next hybrid) already dominates most of the win for free.

---

## 1. Disambiguating "cache eviction"

The three meanings, and what a training-time intervention can move:

**(a) Within-sequence token eviction under a KV budget** (H2O [1], SnapKV [2], StreamingLLM [3], TOVA [4] style). These are *inference-time, training-free* policies that drop KV entries when the cache exceeds a budget. A training-time intervention can move this in two ways: (i) make the whole cache small enough that no budget is exceeded (architectural compression — GQA/MLA/CLA/SWA/SSM), or (ii) train the model to *tolerate* a specific eviction policy (see §2, eviction-aware training — thin literature, mostly 2025–26). **Relevant to Studio only marginally**: Studio launches llama-server with `--no-context-shift` when supported (`studio/backend/core/inference/llama_cpp.py`, ~line 15980), so llama.cpp's StreamingLLM-style context shifting is disabled and overflow fails rather than evicts. There is no H2O/SnapKV in llama.cpp mainline. So on this repo's actual serving path, (a) essentially does not occur.

**(b) Prefix/prompt-cache block eviction across requests** (vLLM APC, SGLang RadixAttention, llama-server per-slot prompt cache + `--cache-reuse`). This is a *serving-capacity* problem: total KV memory can't hold all live conversations, so cached prefixes get recomputed. Training can move it only *indirectly* — a model with 4× smaller KV per token holds 4× more cached prefix in the same memory, raising hit rate. Training cannot move hit rate *directly*: hit rate is determined by prompt-prefix identity (tokenization, chat-template stability, request routing), which is a serving problem, not a weights problem (see §2 last bullet). **On Studio this pressure is minimal**: single user, default 1 parallel slot (`llama_server_args.py` owns `-np/--parallel`; `LoadRequest.n_parallel` optional), one or two conversations. Switching conversations can trash the slot's prompt cache and force a re-prefill, but that costs prefill *compute*, and no finetune changes prefill FLOPs except SwiftKV-style layer skipping (blocked in llama.cpp, §4).

**(c) Hardware bandwidth: KV cache as the decode bottleneck.** Every decode step streams the *entire* weight set plus the KV cache up to the current position through memory. This is the meaning a training-time intervention genuinely moves — a smaller KV cache is fewer bytes per decode step and less resident memory. But it only dominates once KV size rivals weights size (arithmetic in §4). On Apple Silicon (unified memory, 100–550 GB/s), reducing KV buys (i) long-context decode speed and (ii) headroom to run longer contexts / bigger quants on 16–24 GB machines.

**Interaction**: shrinking per-token KV (c) is the one lever that helps all three: it delays (a)'s budget, raises (b)'s effective cache capacity, and directly cuts (c)'s bytes/step. For this repo — llama.cpp in Studio, single-user desktop — **(c) is the only one that matters, and only at long context or tight memory.** The finetuning side (unsloth on CUDA) is where the conversion would run; it doesn't itself suffer KV pressure during training (training has no KV cache in the inference sense).

## 2. Prior art: short-finetune conversions to a cache-cheaper model

### Attention-shape surgery

- **MHA→MQA/GQA uptraining** — Ainslie et al., *GQA* (arXiv 2305.13245, May 2023, EMNLP 2023) [5]. Mean-pool the KV projection heads of an MHA checkpoint into fewer groups, then "uptrain" with **~5% of original pretraining compute**. Uptrained GQA ≈ MHA quality at ≈ MQA speed (T5-XXL experiments). This is the template every later retrofit follows. Caveat for 2026: nearly every model users run is *already* GQA (Llama 3: 8 KV heads; Qwen 3: 8/4). The live variant is *GQA narrowing* (8→4→2→1), which the DMC paper used as its baseline ("up-trained GQA 2×") — it works but loses measurably more quality per compression step than DMC at 4×+ [8].
- **MHA/GQA→MLA conversion** — *TransMLA* (arXiv 2502.07864, Feb 2025) [6]: converts any GQA model to MLA (DeepSeek-style latent KV); reports **93% KV-cache compression on LLaMA-2-7B, ~10.6× speedup at 8k**, needing **~6B tokens** of finetuning to recover benchmark parity. *MHA2MLA* (arXiv 2502.14837, Feb 2025) [7]: partial-RoPE removal + joint SVD init; reports recovery with **0.3%–0.6% of pretraining data**. Both are the biggest compression ratios in the "still-attention" family.
- **Cross-layer KV sharing** — *CLA* (Brandon et al., arXiv 2405.12981, May 2024) [9]: shares KV across adjacent layers, 2× KV reduction at ~equal quality — but the paper's experiments are **from-scratch pretraining** (1B/3B), not retrofit. *YOCO* (arXiv 2405.05254, May 2024) [10]: decoder-decoder, caches KV once — **from scratch**. A systematic retrofit study exists (arXiv 2410.14442 [11]) showing post-hoc cross-layer sharing needs substantial uptraining to recover. *SwiftKV* (Snowflake, arXiv 2410.03960, Oct 2024) [12] is the practical retrofit: SingleInputKV (later layers' KV computed from an earlier layer's hidden state → ~50% prefill FLOPs cut) + AcrossKV (adjacent-layer KV merging → 62.5% KV memory cut), recovered via distillation on a small SFT-scale dataset training only the affected Q/K/V projections; ~2× serving throughput in vLLM for Llama-3.1-8B/70B with ~1-point benchmark deltas. Note: "LISA-style layer sharing" from the prompt — I could not find a KV-sharing paper by that name; the well-known LISA (arXiv 2403.17919) is layerwise importance sampling for *training memory*, unrelated. Flagged rather than cited.
- **KV head pruning/merging** — ThinK (arXiv 2407.21018) prunes key-cache *channels* at inference; various head-merging works exist. Mostly training-free; a finetune on top follows the GQA-uptraining recipe.

### Hybrid / local-attention conversion

- **Interleaved SWA as a design point**: Gemma 2 (1:1 local:global, 4k window) and Gemma 3 (arXiv 2503.19786 [13], **5:1 local:global, 1024-token window**) show interleaving costs little perplexity and slashes KV at long context — but these are *pretrained that way*.
- **Character.AI's production writeup** [14]: MQA everywhere + local attention on 5 of 6 layers + cross-layer KV sharing ⇒ **>20× KV reduction** "without regressing quality" — also trained-in from scratch, not a retrofit.
- **Retrofit path**: *MixAttention* (Databricks, arXiv 2409.15012 [15]) explores SWA + cross-layer sharing variants and finds retrofit feasibility depends heavily on which layers stay global. Post-hoc SWA-ification of a full-attention checkpoint with a short finetune has no canonical "5%-compute" recipe in the literature I can verify; training-free variants (e.g. NLL-guided layer selection for sliding-window adaptation, arXiv 2606.27791) suggest partial conversion is tolerable but with visible long-context degradation. Treat cost as "unknown, likely ≥ GQA-uptraining scale, with long-context retrieval the failure mode."

### Linear-attention / SSM distillation (removes KV cache; leaves O(1) recurrent state)

- **MOHAWK / Phi-Mamba** (arXiv 2408.10189, Aug 2024, NeurIPS 2024) [16]: three-stage distillation (matrix-mixing match → hidden-state match → end-to-end); Phi-Mamba with **3B tokens**, hybrid with 5B — <1% of from-scratch data. Quality: best non-Transformer at its size *at the time*, but not output-identical to the teacher.
- **Llamba** (arXiv 2502.14458, Feb 2025) [17]: Llama-3.x → Mamba family via MOHAWK at scale; Llamba-8B with **12B tokens** (<0.1% of Llama-3.1's data), "comparable benchmark performance", large batch-throughput wins.
- **The Mamba in the Llama** (arXiv 2408.15237, Aug 2024, NeurIPS 2024) [18]: distills to *hybrid* (keeps 1/4–1/2 attention layers) with **20B tokens**; hybrid ≈ teacher on chat benchmarks.
- **LoLCATs** (arXiv 2410.10254, Oct 2024, ICLR 2025) [19]: linearizes 7–8B models training **0.2% of params on 40M tokens (~5 h on one A100)** — cheapest by far — but closes only **>80% of the linearizing quality gap**, i.e. explicitly *not* output-preserving. RADLADS (arXiv 2505.03005) [20] does it with 350–700M tokens to RWKV-style decoders.
- Fidelity summary: benchmark parity is achievable at 8B with 10–20B tokens; *token-level output agreement* with the teacher is not achieved nor claimed by any of these. If "outputs stay as close as possible" is a hard constraint, full linearization fails it; hybrids come closest.

### KV quantization-aware finetuning

- **KIVI** (arXiv 2402.02750, ICML 2024) [21] and **KVQuant** (arXiv 2401.18079) [22] are the canonical schemes (per-channel keys, per-token values, 2–3 bit) and both are **explicitly tuning-free**. llama.cpp independently ships `--cache-type-k/v` (q8_0, q4_0, iq4_nl, q5_x with FA) — Studio already passes these through (`llama_server_args.py`) and budgets for them (`_kv_bytes_per_elem`). Evidence that a finetune materially improves *KV-cache* quantization (as opposed to weight QAT) is thin; the field's conclusion is that post-hoc per-channel/per-token schemes already sit near the quality ceiling at 4-bit, and 2-bit gains come from better rotations/outlier handling (RotateKV, arXiv 2501.16383), not training. **Verdict: a finetune buys little here; q8_0 KV is effectively free quality-wise and already available in Studio.**

### Eviction-aware training

- **DMC — Dynamic Memory Compression** (NVIDIA, arXiv 2403.09636, Mar 2024, ICML 2024) [8] is the flagship *retrofit-finetune* in this class: continued pretraining on a negligible fraction of the original data teaches the model, per head/layer, to *merge* incoming KV pairs into the cache top instead of appending. Llama-2 7B/13B/70B at up to **4× cache compression with preserved downstream performance, ~3.4–3.9× throughput** on H100, beating up-trained GQA and eviction policies. Public retrofits exist (nvidia/Llama-2-7B-DMC-8x on HF). Catch: DMC needs *runtime support* (a decision variable and merge op in the KV update) — not in llama.cpp and not GGUF-expressible.
- Training a model to tolerate a *fixed external* eviction policy: the recent literature (2025–26) mostly trains a lightweight *predictor* rather than the model — IndexMem (arXiv 2605.25475), learned per-head eviction policies on frozen backbones (arXiv 2602.10238), semantic-aware prefix-cache eviction (arXiv 2605.18825) [23]. I found no strong evidence that finetuning the backbone to tolerate H2O/SnapKV-style dropping beats DMC or simple architectural compression; DMC's comparison suggests learned merge > learned drop.

### Training for prefix-cache hit rate

As predicted: **this is a serving/routing problem, not a training problem.** Hit rate is a function of exact token-prefix reuse (stable system prompts, deterministic chat templates, session affinity, RadixAttention/`--cache-reuse` matching). No credible work trains model weights to raise prefix-cache hit rate; the "learned" work in this area (e.g. semantic-aware prefix-cache eviction [23]) trains the *cache manager*. A model-side finetune can only help indirectly by shrinking KV per token so more prefixes fit (§1b).

## 3. Measuring "keeps the existing output"

Benchmark scores are a backstop, not the metric — a model can hold MMLU while paraphrasing every answer. Measure output preservation directly against the frozen teacher:

1. **Greedy top-1 agreement rate**: fraction of positions where student argmax == teacher argmax, teacher-forced on held-out prompts (mix: chat, code, long-context ≥16k). Report per context-length bucket (2k / 8k / 32k) and per position-in-sequence decile — cache-shape surgery degrades *late* positions first.
2. **Forward KL(teacher ‖ student)** of full next-token distributions, mean and P95 in nats/token, teacher-forced. Reverse KL as a secondary (catches mode-dropping the forward KL hides). Practical note: at 128k vocab, store teacher top-k (k=64–128) + tail mass rather than full logits.
3. **Greedy-decode exact-match prefix length**: generate greedily from both models on the same prompts; record tokens until first divergence, and total edit distance over 512-token generations. This is the metric users actually experience.
4. **Per-position divergence over long free-running generations**: teacher-forced KL measured along *student-generated* trajectories, to detect compounding drift that teacher-forced metrics mask.
5. **Task backstop**: MMLU/GSM8K/HumanEval/RULER (long-context retrieval — this is where SWA/eviction surgery fails first) within noise of the teacher.

**Defensible acceptance bar** (for an "invisible" conversion): ≥95% greedy top-1 agreement at 2k context and ≥90% at 16k; mean forward KL ≤0.05 nats/token (P95 ≤0.5); median divergence-free greedy prefix ≥100 tokens on chat prompts; RULER within 2 points of teacher. What the literature suggests is achievable: GQA-narrowing and DMC-class retrofits with a distillation objective can plausibly meet this (they report ~0 benchmark delta; token-level agreement is rarely published — expect to be the first to publish it). MLA conversion at 90%+ compression: benchmark parity yes, 95% top-1 agreement uncertain. SSM distillation: will **not** meet this bar (LoLCATs closes only ~80% of its gap; Llamba/MambaInLlama claim benchmark parity, not token agreement) — frame those as "new model distilled from X", not "X, but faster".

Reference-model plumbing note: unsloth already runs frozen-reference forward passes for DPO (`unsloth/models/dpo.py`), so the two-model pattern is not foreign to the codebase; a memory-cheaper option is offline teacher top-k logit dumps generated with the same model pre-surgery.

## 4. Feasibility in this repo

### Finetuning path (unsloth core)

- Unsloth's fast paths read attention shape from config, not constants: `unsloth/models/llama.py` (~lines 395–419, 697–705) uses `self.config.num_key_value_heads` / `num_key_value_groups` for both training forward and its paged inference cache. **GQA narrowing is therefore expressible**: do checkpoint surgery *before* `FastLanguageModel.from_pretrained` (mean-pool `k_proj`/`v_proj` head groups per the GQA paper, edit `num_key_value_heads` in config) and every patched kernel — `kernels/fast_lora.py` `LoRA_QKV` is shape-agnostic, RoPE/CE/RMSNorm kernels don't care — keeps working. Full-finetune of k/v/q projections + LoRA elsewhere also fits (`assert n_kv_heads * n_groups == n_heads` at llama.py:700 holds for any divisor).
- **MLA conversion is not expressible** in the fast path: it adds latent down/up projections and decoupled-RoPE — no longer a Llama-config model, so it falls off `FastLlamaModel` patching entirely (DeepSeek-V2-style archs load via the generic/vision path without the hand-written attention forwards). Doable, but you'd be writing a new model integration, not a recipe.
- **Cross-layer KV sharing** likewise has no HF-Transformers-native config for Llama; it needs custom modeling code that bypasses unsloth's per-layer patching.
- **SSM hybrids**: unsloth already finetunes Falcon-H1 (`unsloth/models/falcon_h1.py`), so *training* a hybrid is supported — but MOHAWK-style distillation *into* one is a multi-stage research pipeline, not present.
- **Distillation loop: does not exist.** The Triton CE kernel (`unsloth/kernels/cross_entropy_loss.py`) is hard-label only (loads a single `label_idx` per row; no soft-target/KL variant). Studio's trainer is TRL `SFTTrainer` (`studio/backend/core/training/trainer.py:84,4193`). What needs building: a KD trainer (chunked KL over teacher top-k logits to avoid materializing [seq × 128k] float tensors), teacher-logit capture (offline dump or paired forward), and the §3 eval harness. All buildable in plain PyTorch first; a fused soft-target CE kernel is an optimization, not a prerequisite.

### Inference path (Studio / llama.cpp) — the binding constraint

Studio serves GGUF via pinned prebuilt llama-server (`studio/install_llama_prebuilt.py`, macOS fallback pin `b9415`, tracking recent upstream). llama.cpp implements architectures as per-arch C++ — a converted model must *exactly match an existing arch's layout*. Expressibility today:

| Technique | GGUF/llama.cpp today? | Evidence |
|---|---|---|
| GQA narrowing (fewer `n_kv_heads`) | **Yes, zero work** — plain llama-arch metadata; `convert_hf_to_gguf` (already vendored via `unsloth/save.py`) handles it | llama.cpp reads `attention.head_count_kv` per model |
| MLA | **Only as DeepSeek-family archs** (PRs #12801, #12227 [24]); a TransMLA-converted Llama is not a valid `deepseek2` checkpoint and has no converter support → upstream work | Studio's own KV estimator has an MLA path keyed on `kv_lora_rank` (`llama_cpp.py:3805`) — for DeepSeek archs |
| Cross-layer KV sharing | **Only Gemma-3n/Gemma-4 archs** (`shared_kv_layers`, `llama_cpp.py:3814-3816`); no generic mechanism. SwiftKV explicitly requested and unimplemented (llama.cpp issue #11415 [25]) | |
| Interleaved SWA retrofit | **Per-arch**: gemma2/gemma3/phi3/cohere2 etc. have SWA patterns; arch `llama` does not (to my knowledge llama.cpp never ran SWA for Mistral-v0.1 llama-arch GGUFs — verify before relying on this) → upstream work | Studio estimator: `sliding_window_pattern`, `kv_*_length_swa` fields |
| SSM / hybrid | **Yes for existing archs** (mamba2, jamba, falcon-h1, nemotron-h, qwen3next — Studio's estimator prices all of these, incl. `kda_head_dim`), but a *distilled* model must land exactly in one of those layouts; Llamba's layout is not upstream | `llama_cpp.py` `_TARGET_KV_EXCLUDES_NEXTN_ARCHS`, `_ssm_*` fields |
| DMC | **No** — needs a runtime merge op in the KV update; no GGUF representation | |
| KV-quant finetune | Moot — llama.cpp cache types are post-hoc and Studio already passes them through | `llama_server_args.py` `_CACHE_TYPE_*_FLAGS` |

Secondary path: Studio also has an MLX backend (`studio/backend/core/inference/mlx_inference.py`), where Python-level architecture flexibility is much higher — a plausible escape hatch for non-GGUF-expressible experiments on Apple Silicon, but it is the alternate path, not the default.

### Is KV even the binding constraint on the Studio target? (arithmetic)

Llama-3.1-8B: 32 layers × 8 KV heads × 128 head-dim. KV bytes/token = 2(K,V) × 32 × 8 × 128 × 2 B (f16) = **128 KiB/token**.

| Context | KV f16 | KV q8_0 (~1.06 B/el) | KV after 8→2 heads, f16 | 8→2 + q8_0 |
|---|---|---|---|---|
| 2k | 0.25 GiB | 0.13 GiB | 0.06 GiB | 0.03 GiB |
| 8k | 1.0 GiB | 0.53 GiB | 0.25 GiB | 0.13 GiB |
| 32k | **4.0 GiB** | 2.1 GiB | 1.0 GiB | 0.53 GiB |

Weights: 8B @ Q4_K_M ≈ 4.9 GB. Decode step ≈ stream weights + KV-so-far: at 2k context that's 4.9 + 0.25 = 5.2 GB/token (KV is 5% — narrowing buys ~nothing); at 32k f16 it's 8.9 GB/token. On an M3 Pro (~150 GB/s): ~17 tok/s → ~30 tok/s ceiling if KV were free; realistically 8→2+q8_0 at 32k ≈ 5.4 GB/token ≈ **~35–40% decode speedup at 32k, ~0% at 2k**. The crossover where KV read equals weights read is ~38k tokens (f16) / ~72k (q8_0). **The bigger practical win is capacity**: on a 16 GB Mac (usable ~10–11 GB for the model), weights 4.9 GB + f16 KV caps context near 40k and competes with the OS; 8→2+q8_0 makes 32k cost 0.5 GiB — long context stops being a memory event. On the multi-user/prefix-cache axis (§1b) Studio's 1-slot default means there is essentially nothing to win.

## 5. Verdict and proposal

**Primary: teacher-consistency distillation harness + GQA KV-head narrowing recipe.** It is the only surveyed transform that today's Studio llama.cpp loads with zero upstream work, it is fully expressible through unsloth's existing fast paths (config-driven `num_key_value_heads`), and the GQA-uptraining/DMC literature says 2–4× KV reduction with ~no benchmark delta is attainable at a few billion tokens or less (start with LoRA-on-all + full-tune of k/v projections, KL-to-teacher objective; escalate data until the §3 bar is met).

Incremental landing plan:
1. **Ships first (smallest useful slice): the fidelity eval harness** — given any (teacher GGUF/HF, student) pair: greedy top-1 agreement by context bucket, forward/reverse KL on teacher top-k, divergence-prefix length, RULER backstop. This is useful *immediately* even with no training feature (it can grade quant choices and third-party models) and it is the referee for everything after.
2. `kv_head_narrow()` checkpoint-surgery utility (mean-pool per GQA paper) + KD trainer (chunked top-k KL; reuse the DPO reference-model plumbing or offline teacher dumps).
3. End-to-end recipe: narrow → distill → `save_pretrained_gguf` → verify in llama-server → publish harness numbers. Target demo: Llama-3.1-8B or Qwen3-8B, 8→2 KV heads, ≥95%/≥90% top-1 agreement at 2k/16k.

**Fallback: MLA conversion (TransMLA/MHA2MLA-style).** ~3× more compression than narrowing and llama.cpp already has strong MLA *kernels* — but it is gated on upstream work (a converter + arch mapping for "MLA-ized Llama", or emitting a legitimate DeepSeek-V2-Lite-shaped checkpoint) and on a new unsloth model integration. Scope it as: prototype on the MLX/PyTorch path first, upstream GGUF support second. Do not start here.

**Risks**: (i) token-level agreement at the §3 bar is stricter than anything the cited papers report — the recipe may need more tokens than the optimistic 0.3–0.6% figures; (ii) long-context agreement (where the win lives) is exactly where narrowed-KV models degrade first; (iii) per-model conversion cost (~B tokens ≈ multi-hundred-GPU-hours at 8B) must be paid by whoever publishes the converted checkpoint — this is a recipe + published-artifacts play, not a button every user presses.

**The honest case against**: (1) every modern target is already GQA — narrowing an 8-KV-head model buys at most 4× on a cache that q8_0 (already in Studio, free, near-lossless) halves anyway; (2) below ~16k context on the Studio target the KV cache is not the bottleneck — weights are — so most sessions see no speedup; (3) the ecosystem increasingly ships cache-cheap models natively (Gemma 3's 5:1 SWA, Qwen3-Next/Nemotron-H hybrids, DeepSeek MLA — all already load in Studio's llama.cpp and are priced by Studio's own KV estimator), so "pick a better model" often dominates "convert this one"; (4) prefix-cache eviction, the headline word in the pitch, is a server-fleet problem Studio's single-slot desktop barely has. The narrower version that retains clear value regardless: **the distillation trainer + output-fidelity harness** — unsloth has neither, both are reusable for quant grading, pruning, and any future conversion, and they are the prerequisite for ever making the "same outputs, cheaper cache" claim honestly.

## References

1. H2O: Heavy-Hitter Oracle for KV Cache — https://arxiv.org/abs/2306.14048
2. SnapKV — https://arxiv.org/abs/2404.14469
3. StreamingLLM (attention sinks) — https://arxiv.org/abs/2309.17453
4. TOVA: Transformers are Multi-State RNNs — https://arxiv.org/abs/2401.06104
5. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — https://arxiv.org/abs/2305.13245
6. TransMLA: Multi-Head Latent Attention Is All You Need — https://arxiv.org/abs/2502.07864
7. MHA2MLA: Towards Economical Inference — https://arxiv.org/abs/2502.14837
8. Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference — https://arxiv.org/abs/2403.09636 ; retrofit checkpoint: https://huggingface.co/nvidia/Llama-2-7B-DMC-8x
9. Reducing Transformer KV Cache Size with Cross-Layer Attention (CLA) — https://arxiv.org/abs/2405.12981
10. You Only Cache Once (YOCO) — https://arxiv.org/abs/2405.05254
11. A Systematic Study of Cross-Layer KV Sharing — https://arxiv.org/abs/2410.14442
12. SwiftKV: Fast Prefill-Optimized Inference with Knowledge-Preserving Model Transformation — https://arxiv.org/abs/2410.03960
13. Gemma 3 Technical Report — https://arxiv.org/abs/2503.19786
14. Optimizing AI Inference at Character.AI — https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/
15. MixAttention (Databricks) — https://arxiv.org/abs/2409.15012
16. MOHAWK / Phi-Mamba: Transformers to SSMs — https://arxiv.org/abs/2408.10189 ; https://github.com/goombalab/phi-mamba
17. Llamba: Scaling Distilled Recurrent Models — https://arxiv.org/abs/2502.14458
18. The Mamba in the Llama — https://arxiv.org/abs/2408.15237
19. LoLCATs: On Low-Rank Linearizing of LLMs — https://arxiv.org/abs/2410.10254 ; https://www.together.ai/blog/linearizing-llms-with-lolcats
20. RADLADS — https://arxiv.org/abs/2505.03005
21. KIVI: Tuning-Free Asymmetric 2-bit KV Quantization — https://arxiv.org/abs/2402.02750
22. KVQuant — https://arxiv.org/abs/2401.18079
23. Learned eviction (2025–26): IndexMem — https://arxiv.org/abs/2605.25475 ; Learning to Evict — https://arxiv.org/abs/2602.10238 ; Semantic-aware prefix-cache eviction — https://arxiv.org/abs/2605.18825
24. llama.cpp MLA: PR #12801 — https://github.com/ggml-org/llama.cpp/pull/12801 ; PR #12227 — https://github.com/ggml-org/llama.cpp/pull/12227
25. llama.cpp SwiftKV feature request (open, unimplemented) — https://github.com/ggml-org/llama.cpp/issues/11415
26. RotateKV (2-bit KV via rotations, training-free) — https://arxiv.org/abs/2501.16383
27. ThinK: key-cache channel pruning — https://arxiv.org/abs/2407.21018

Repo evidence paths: `unsloth/models/llama.py` (config-driven `n_kv_heads`), `unsloth/kernels/cross_entropy_loss.py` (hard-label-only CE), `unsloth/kernels/fast_lora.py` (shape-agnostic LoRA_QKV), `unsloth/models/falcon_h1.py` (hybrid SSM finetune support), `unsloth/save.py` (GGUF export), `studio/backend/core/inference/llama_cpp.py` (5-path KV estimation: full/SWA/MLA/SSM/shared-KV; `--no-context-shift`), `studio/backend/core/inference/llama_server_args.py` (cache-type passthrough, slot ownership), `studio/install_llama_prebuilt.py` (llama.cpp pin), `studio/backend/core/training/trainer.py` (SFTTrainer, no KD), `studio/backend/core/inference/mlx_inference.py` (MLX alternate path).

Uncertainty flags: "LISA-style KV layer sharing" could not be verified as an existing paper (the known LISA is unrelated); llama.cpp's non-support of SWA under arch `llama` (Mistral v0.1) is from memory and should be verified against the pinned tag; SwiftKV's exact token budget is not restated here because I could not re-verify the number.
