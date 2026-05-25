"""Speculative decoding from scratch.

This is the algorithm from Leviathan et al. 2022 ("Fast Inference from
Transformers via Speculative Decoding"). The implementation is meant to
teach, not to be the fastest possible — production speculative decoding
uses much more aggressive batching and KV-cache tricks.

The algorithm:
1. Draft model proposes k tokens autoregressively.
2. Target model evaluates the k+1 logits for [prompt + draft_tokens] in
   ONE forward pass (this is the speedup — k+1 tokens for the cost of 1).
3. For each of the k drafted tokens, accept it with probability
   min(1, p_target(t) / p_draft(t)). Reject and resample on the first
   rejection.
4. If all k accepted: also sample from the (k+1)th target distribution
   (free bonus token).
5. If some rejected: sample one token from the corrected distribution
   (target - draft, clipped) at the rejection point.

This guarantees the output distribution matches sampling from the target
model directly. So spec decoding doesn't change quality, only latency.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)


@dataclass
class SpecResult:
    tokens: list[int]
    accepted_total: int
    proposed_total: int
    decode_rounds: int  # how many spec rounds we ran

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_total / max(self.proposed_total, 1)


@torch.inference_mode()
def speculative_decode(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    k: int = 4,
    temperature: float = 1.0,
    device: str | None = None,
) -> SpecResult:
    """Generate up to max_new_tokens using speculative decoding.

    target_model, draft_model: HF causal LMs sharing the same vocab.
    k: number of draft tokens per round. The classic tuning parameter.
    """
    device = device or next(target_model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    generated = input_ids
    accepted_total = 0
    proposed_total = 0
    rounds = 0

    while generated.shape[1] - prompt_len < max_new_tokens:
        rounds += 1
        # 1) Draft model proposes k tokens autoregressively.
        draft_tokens, draft_probs = _draft_propose(
            draft_model, generated, k=k, temperature=temperature
        )
        # draft_tokens: (k,) ; draft_probs: (k, vocab) — prob distribution at each step

        # 2) Target model evaluates all k+1 positions in one forward pass.
        candidate = torch.cat([generated, draft_tokens.unsqueeze(0)], dim=1)
        target_logits = target_model(candidate).logits  # (1, prompt_len + N + k, vocab)
        target_probs = _softmax_temp(
            target_logits[0, -(k + 1):, :], temperature
        )  # (k+1, vocab)

        # 3) Acceptance loop
        new_tokens = []
        for i in range(k):
            t = draft_tokens[i].item()
            p_t = target_probs[i, t].item()
            q_t = draft_probs[i, t].item()
            proposed_total += 1
            if torch.rand(1).item() < min(1.0, p_t / max(q_t, 1e-9)):
                # accept
                new_tokens.append(t)
                accepted_total += 1
            else:
                # reject -> sample corrected distribution
                corrected = (target_probs[i] - draft_probs[i]).clamp(min=0)
                corrected = corrected / corrected.sum().clamp(min=1e-9)
                t_new = torch.multinomial(corrected, 1).item()
                new_tokens.append(t_new)
                break
        else:
            # All k accepted -> bonus token from the (k+1)th target distribution
            t_bonus = torch.multinomial(target_probs[k], 1).item()
            new_tokens.append(t_bonus)

        # Append accepted (and possibly bonus or corrected) tokens
        new_tokens_t = torch.tensor([new_tokens], device=device, dtype=generated.dtype)
        generated = torch.cat([generated, new_tokens_t], dim=1)

        # EOS check
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in new_tokens:
            break

    new_only = generated[0, prompt_len:].tolist()[:max_new_tokens]
    return SpecResult(
        tokens=new_only,
        accepted_total=accepted_total,
        proposed_total=proposed_total,
        decode_rounds=rounds,
    )


def _draft_propose(
    draft_model,
    context: torch.Tensor,
    k: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the draft model autoregressively for k steps. Return tokens and
    the probability distribution at each step (needed for the acceptance test).
    """
    seq = context
    tokens = []
    probs_list = []
    for _ in range(k):
        logits = draft_model(seq).logits[0, -1, :]
        probs = _softmax_temp(logits.unsqueeze(0), temperature).squeeze(0)
        t = torch.multinomial(probs, 1)
        tokens.append(t)
        probs_list.append(probs)
        seq = torch.cat([seq, t.unsqueeze(0)], dim=1)
    tokens_t = torch.stack(tokens).squeeze(-1)  # (k,)
    probs_t = torch.stack(probs_list, dim=0)  # (k, vocab)
    return tokens_t, probs_t


def _softmax_temp(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        # greedy: degenerate to one-hot
        idx = torch.argmax(logits, dim=-1, keepdim=True)
        out = torch.zeros_like(logits)
        out.scatter_(-1, idx, 1.0)
        return out
    return torch.softmax(logits / temperature, dim=-1)
