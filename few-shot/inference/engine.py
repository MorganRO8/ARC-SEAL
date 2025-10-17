from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
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
    enforce_eager: bool = False,
    enable_lora: bool = True,
    max_lora_rank: int = 64,
    quantization: Optional[str] = None,
    lora_repo: Optional[str] = None,
    lora_target_modules: Optional[List[str]] = None,
    dtype: Optional[str] = "float16",
    max_model_len: int = 8192,
    max_num_batched_tokens: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = 0.9,
    tensor_parallel_size: Optional[int] = 1,
) -> LLMEngine:
    """Initialize the LLMEngine."""

    engine_args_kwargs = dict(
        model=model,
        enable_lora=enable_lora,
        max_lora_rank=max_lora_rank,
        enforce_eager=enforce_eager,
        quantization=quantization,
        load_format="bitsandbytes" if quantization else "auto",
        max_model_len=max_model_len,
    )

    if dtype:
        engine_args_kwargs["dtype"] = dtype

    if lora_target_modules is not None:
        engine_args_kwargs["lora_target_modules"] = lora_target_modules

    if max_num_batched_tokens is not None:
        engine_args_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

    if gpu_memory_utilization is not None:
        engine_args_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    if tensor_parallel_size is not None:
        engine_args_kwargs["tensor_parallel_size"] = tensor_parallel_size

    engine_args = EngineArgs(**engine_args_kwargs)

    return LLMEngine.from_engine_args(engine_args)


def process_requests(
    engine: LLMEngine, test_prompts: List[Tuple[str, SamplingParams, Optional[LoRARequest], str]]
) -> Dict[str, List[str]]:
    """Continuously process a list of prompts and handle the outputs."""
    all_outputs: Dict[str, List[str]] = {}
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_param, lora_request, idx = test_prompts.pop(0)
            find_start = prompt.find("<|begin_of_text|>") + len("<|begin_of_text|>")
            prompt = prompt[find_start:]
            engine.add_request(idx, prompt, sampling_param, lora_request=lora_request)

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                texts = [output.text for output in request_output.outputs]
                all_outputs[str(request_output.request_id)] = texts
    
    return all_outputs
