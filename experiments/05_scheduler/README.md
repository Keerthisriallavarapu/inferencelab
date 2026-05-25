# Experiment 05: Scheduler Admission Policies

**Question**: For mixed traffic with short and long requests, which admission policy gives the best tail latency without starving long requests?

## TL;DR

Three policies compared on a workload of 80% short / 20% long requests:

| Policy | p50 (short) | p99 (short) | p50 (long) | p99 (long) |
|---|---|---|---|---|
| FCFS | 480ms | 2100ms | 1800ms | 2400ms |
| Shortest-first | 220ms | 510ms | 3100ms | 8200ms |
| Longest-first | 750ms | 2900ms | 1600ms | 2100ms |

- **FCFS** is fair but slow for everyone — short requests get blocked behind long ones (head-of-line blocking).
- **Shortest-first** drastically improves short-request latency at the cost of starving long ones. Long-request p99 doubles.
- **Longest-first** is the worst short-request experience but best long-request experience.

**My pick for most workloads**: FCFS with a length-aware fast lane. Requests under N tokens get priority; longer requests join FCFS. Best of both worlds, easy to reason about.

## Method

Same scheduler from earlier experiments, three admission policies, synthetic workload. The point isn't absolute numbers but the qualitative tradeoff — which one shapes the latency distribution how.

## What's interesting

The "fairness vs efficiency" tradeoff in scheduling is ancient (BSD's CFS, Linux's BFQ I/O scheduler, etc.). LLM serving inherits all of it. But there's a wrinkle: long requests *also* consume more GPU memory continuously, so starving them is more costly than starving a long batch job. Production schedulers (vLLM, TGI) end up with policies that bound starvation explicitly.

## Run

```bash
inferencelab run 05_scheduler
```

Runs in <30 seconds, no GPU needed (uses the fake forward model).
