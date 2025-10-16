import asyncio
import itertools
from collections.abc import Iterable, Iterator
from typing import Any, Callable, Dict, List, Optional, Union


MESSAGE = Dict[str, Union[str, Dict]]
MESSAGES = List[MESSAGE]


class LocalLMClient:
    """Simple wrapper around a locally hosted language model."""

    def __init__(
        self,
        model_path: str,
        backend: str = "vllm",
        sampling_params: Optional[Any] = None,
        **backend_kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.backend = backend.lower()
        self._sampling_params = sampling_params

        if self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:  # pragma: no cover - import error branch
                raise ImportError(
                    "vllm is required for the 'vllm' backend"
                ) from exc

            backend_kwargs.setdefault("dtype", "float16")
            backend_kwargs.setdefault("max_model_len", 4096)
            backend_kwargs.setdefault("max_num_batched_tokens", 4096)
            backend_kwargs.setdefault("gpu_memory_utilization", 0.6)
            if backend_kwargs.get("enforce_eager") is None:
                backend_kwargs["enforce_eager"] = True

            self._sampling_params = sampling_params or SamplingParams()
            self._model = LLM(model=model_path, **backend_kwargs)
            self._generate = self._generate_vllm

        elif self.backend == "transformers":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - import error branch
                raise ImportError(
                    "transformers is required for the 'transformers' backend"
                ) from exc

            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(model_path, **backend_kwargs)
            device = backend_kwargs.get("device")
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device)
            self._generate = self._generate_transformers

        else:  # pragma: no cover - unsupported backend
            raise ValueError(f"Unsupported backend: {backend}")

    def generate(self, prompts: Union[str, List[str]], **generate_kwargs: Any) -> Any:
        """Generate model outputs for one or more prompts."""

        if isinstance(prompts, str):
            prompts = [prompts]

        return self._generate(prompts, **generate_kwargs)

    def _generate_vllm(self, prompts: List[str], **generate_kwargs: Any) -> Any:
        from vllm import SamplingParams

        sampling_params = generate_kwargs.pop("sampling_params", None)
        if sampling_params is None:
            if self._sampling_params is not None:
                sampling_params = self._sampling_params
            else:
                sampling_params = SamplingParams(**generate_kwargs)
                generate_kwargs = {}

        return self._model.generate(prompts, sampling_params=sampling_params, **generate_kwargs)

    def _generate_transformers(self, prompts: List[str], **generate_kwargs: Any) -> List[str]:
        import torch

        tokenizer_kwargs = generate_kwargs.pop("tokenizer_kwargs", {})
        encoded = self._tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            **tokenizer_kwargs,
        ).to(self._model.device)

        with torch.no_grad():
            generations = self._model.generate(**encoded, **generate_kwargs)

        sequences = generations
        if hasattr(generations, "sequences"):
            sequences = generations.sequences

        return [
            self._tokenizer.decode(sequence, skip_special_tokens=True)
            for sequence in sequences
        ]


def batch(inputs: Union[List, Iterable], n: int):
    "Batch data into iterators of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")

    if not isinstance(inputs, Iterator):
        inputs = iter(inputs)

    while True:
        chunk_it = itertools.islice(inputs, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return

async def run_agent(
    inputs: List[MESSAGES],
    async_llm: Callable[[MESSAGES], Any],
    batch_size: int = 20,
) -> Any:
    """Runs the API function in parallel on the inputs."""
    for task_inputs in batch(inputs, batch_size):
        tasks = [async_llm(task_input) for task_input in task_inputs]

        results = await asyncio.gather(*tasks, return_exceptions=False)

        for result in results:
            yield result

