# What I learned from 5 experiments in LLM inference

I spent about two months running a series of inference-optimization experiments to understand what actually matters when serving LLMs. This is the synthesis. Specific numbers are in each experiment's README.

## The cliff notes

1. **Speculative decoding is the single biggest win for low-concurrency serving.** ~1.5-2x throughput, no quality hit. The "free lunch" everyone says it is. Tuning k matters; default of 4-5 is usually right.

2. **Continuous batching matters most when prompt-length variance is high.** Uniform workloads don't benefit much. Real production workloads always have variance, so CB is essentially required.

3. **PagedAttention's headline benefit isn't speed; it's memory efficiency that lets you batch more.** Same idea as a malloc replacement: faster individual operations are nice, but the real win is being able to allocate things you couldn't before.

4. **INT8 quantization is essentially free.** I cannot find a reason not to use it. INT4 has real tradeoffs that depend on your task.

5. **Scheduling is where the production smarts hide.** Most serving frameworks ship with FCFS by default. For real workloads, smarter admission policies (or fast lanes for short requests) matter more than I expected.

## The thing I was wrong about

Going in, I thought the order of importance for serving optimization was: model architecture → quantization → batching → scheduling. After running these experiments I'd reverse it. The base model is mostly fixed (you ship what you trained). Quantization is one knob with a clear answer (INT8 always; INT4 if you need it). The real production wins are in batching and scheduling — and they compound, while individual model tricks don't.

This is also where the engineering work is most painful, which is why frameworks like vLLM and TGI are worth their weight. Building your own scheduler is doable; building one as good as vLLM's takes a team.

## What I'd build into a serving stack today

If I were starting a serving system for an internal product:

1. **Use vLLM or TGI.** Don't reinvent. Use this lab to understand *why* they're built the way they are.
2. **Speculative decoding for low-concurrency endpoints** (e.g. interactive chat with one user per request stream).
3. **Quantize to INT8 by default.** Test INT4 on your actual evals before shipping it for hard tasks.
4. **Monitor batch fill rate.** If your max_batch_size is rarely reached, your admission policy isn't keeping up. Reduce it or fix the policy.
5. **Length-aware fast lane.** Most production traffic is heavily skewed toward short requests. Letting them bypass long ones is a huge UX win and almost free to implement.

## Caveats on all of this

These experiments were run on a single RTX 4090 in my home office. Some patterns will differ at scale:

- **Multi-GPU**: tensor parallelism changes the cost model significantly. Spec decoding can be tricky to apply with TP. CB still wins.
- **Real workloads**: my synthetic prompts are obviously not real traffic. The shape of conclusions should hold, the exact numbers will not.
- **Specific models**: I used Llama 3.1 8B. Larger models amplify some effects (memory pressure makes paging more important) and shrink others (quantization quality drops more on smaller models, paradoxically — large models are more redundant).

The point of running these myself wasn't to produce numbers anyone should trust. It was to internalize the *shape* of the tradeoffs so I can read a paper or evaluate a serving framework with a sense of where it lands in the design space.

That part worked.
