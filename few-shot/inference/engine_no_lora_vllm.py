from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer
from vllm import LLM, EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def get_sampling_params(
    tokenizer: PreTrainedTokenizer,
    num_tokens: int,
    max_tokens: int,
    temperature: float = 0.0,
    n: int = 1,
) -> SamplingParams:
    max_new_tokens = max_tokens - num_tokens
    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=n,
        stop=[tokenizer.eos_token, "<|eot_id|>"],
        # best_of=10,
        # use_beam_search=True,
    )


def initialize_engine(
    model: str,
    enforce_eager: bool = True,
    enable_lora: bool = True,
    max_lora_rank: int = 64,
    quantization: Optional[str] = None,
    lora_repo: Optional[str] = None,
    lora_target_modules: Optional[List[str]] = None,
    dtype: str = "float16",
    max_model_len: int = 4096,
    max_num_batched_tokens: int = 4096,
    gpu_memory_utilization: float = 0.6,
    tensor_parallel_size: int = 1,
) -> LLMEngine:
    """Initialize the LLMEngine."""

    llm_kwargs = {
        "model": model,
        "enable_lora": False,
        "max_lora_rank": max_lora_rank,
        "enforce_eager": enforce_eager,
    }

    if dtype:
        llm_kwargs["dtype"] = dtype
    if max_model_len and max_model_len > 0:
        llm_kwargs["max_model_len"] = max_model_len
    if max_num_batched_tokens and max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if gpu_memory_utilization and gpu_memory_utilization > 0:
        llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    if tensor_parallel_size and tensor_parallel_size > 0:
        llm_kwargs["tensor_parallel_size"] = tensor_parallel_size

    llm = LLM(**llm_kwargs)

    return llm

def process_requests(
    engine: LLMEngine, test_prompts: List[Tuple[str, SamplingParams, Optional[LoRARequest], str]]
) -> Dict[str, List[str]]:
    """Continuously process a list of prompts and handle the outputs."""
    all_outputs: Dict[str, List[str]] = {}
    while test_prompts:
        prompt, sampling_param, lora_request, idx = test_prompts.pop(0)
        find_start = prompt.find("<|begin_of_text|>") + len("<|begin_of_text|>")
        prompt = prompt[find_start:]
        request_outputs = engine.generate(prompt, sampling_param)

        for request_output in request_outputs:
            if request_output.finished:
                texts = [output.text for output in request_output.outputs]
                all_outputs[idx] = texts
    return all_outputs
