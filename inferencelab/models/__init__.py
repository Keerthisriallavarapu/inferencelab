"""Model loading helpers.

We keep loading separate from the benchmark code so we can swap models in
experiments without rewriting the benchmark itself. Each loader returns a
small interface (a callable that generates given prompts).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    tokens_per_second: float


class HFModelWrapper:
    """Thin wrapper around a HuggingFace causal LM for benchmarking.

    We don't use pipeline() because it does extra preprocessing that adds
    measurement noise. Tokenize once, generate, measure.
    """

    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device: str = "auto",
        quantization: str | None = None,  # "int8" | "int4" | None
        attn_implementation: str = "sdpa",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("HFModelWrapper requires `pip install -e .[gpu]`") from e

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": device,
            "attn_implementation": attn_implementation,
        }
        if quantization == "int8":
            kwargs["load_in_8bit"] = True
        elif quantization == "int4":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        log.info("Loading %s (dtype=%s, quant=%s)", model_id, dtype, quantization)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.model_id = model_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> GenerationResult:
        import time as _time
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_token_count = inputs.input_ids.shape[1]

        # CUDA sync to make timing accurate
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = _time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = _time.perf_counter() - start

        new_tokens = outputs[0, input_token_count:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        n_out = new_tokens.shape[0]

        return GenerationResult(
            text=text,
            input_tokens=input_token_count,
            output_tokens=n_out,
            elapsed_seconds=elapsed,
            tokens_per_second=(n_out / elapsed) if elapsed > 0 else 0.0,
        )
