# SEAL - few-shot

This document contains the commands to reproduce the SEAL Few Shot Experiments on ARC. 
Code is adopted from: [Ekin's Repo](https://github.com/ekinakyurek/marc/tree/main)

## SEAL RL Iteration 1

### Environment setup

Create a fresh virtual environment (``python -m venv .venv``) and install the pinned dependencies from the repository root before running any of the commands below:

```bash
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # run from the repository root
```

The dependency pins intentionally keep ``numpy`` on the 1.26 series so that the ``scikit-learn`` wheels pulled in by ``transformers`` remain binary compatible.  Installing the requirements inside a clean environment avoids the ``ValueError: numpy.dtype size changed`` crash that can arise when an older ``scikit-learn`` build from a base Conda installation is left alongside ``numpy`` 2.x.

### 1. Training on 12 Problems (Iteration 1)

Train the base model on 12 problems from ARC train set:

```bash
python self-edit.py \
    --experiment_name=training_set_iteration_1 \
    --challenge_file=${DATA_DIR}/arc-agi_training_challenges_filtered_1B_training_set.json \
    --solution_file=${DATA_DIR}/arc-agi_training_solutions_filtered_1B_training_set.json \
    --model_name=meta-llama/Llama-3.2-1B-Instruct \
    --n_tasks=12 \
    --n_self_edits_per_task=15
```

### 2. Evaluation on Iteration 1 LoRAs

Evaluate the trained LoRAs from iteration 1:

```bash
python eval-self-edits.py \
    --experiment_folder=${TTI_DIR}/training_set_iteration_1 \
    --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct \
    --lora_checkpoints_folder=${LORA_DIR}/self-edit/training_set_iteration_1 \
    --temperature=0 \
    --n_sample=1 \
    --data_file=${DATA_DIR}/arc-agi_training_challenges_filtered_1B_training_set.json \
    --solution_file=${DATA_DIR}/arc-agi_training_solutions_filtered_1B_training_set.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format \
    --num_examples=11 \
    --n_self_edits=15
```

### 3. RestEM on Iteration 1 (8 Epochs)

Run RestEM training for 8 epochs:

```bash
python BC-self-edit.py \
    --configs_and_indices=${LORA_DIR}/self-edit/training_set_iteration_1/final_configs_and_indices.json \
    --results=${LORA_DIR}/self-edit/training_set_iteration_1/final_results.json \
    --model_name=meta-llama/Llama-3.2-1B-Instruct \
    --lora_rank=16 \
    --lora_alpha=16 \
    --num_train_epochs=8 \
    --per_device_train_batch_size=5 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-5
```

### 4. Create Self-Edits on Eval Set (RL Iteration 1, 8 Epochs)

Generate self-edits on evaluation set using the 8-epoch RL model:

```bash
python self-edit.py \
    --experiment_name=eval_RL_iteration_1_8_epoch \
    --challenge_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json \
    --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json \
    --model_name=${LORA_DIR}/self-edit/training_set_iteration_1/RL_trained_model_iteration_1_8_epoch \
    --n_tasks=10 \
    --n_self_edits_per_task=5
```

### 5. Evaluate RL Iteration 1 (8 Epochs) on Eval Set

Evaluate the 8-epoch RL model on the evaluation set:

```bash
python eval-self-edits.py \
    --experiment_folder=${TTI_DIR}/eval_set_RL_iteration_1_8_epoch \
    --pretrained_checkpoint=${LORA_DIR}/self-edit/training_set_iteration_1/RL_trained_model_iteration_1_8_epoch \
    --lora_checkpoints_folder=${LORA_DIR}/self-edit/eval_RL_iteration_1_8_epoch \
    --temperature=0 \
    --n_sample=1 \
    --data_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json \
    --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format \
    --num_examples=9 \
    --n_self_edits=5
```

## Baseline Evaluation

### Evaluate Baseline Model

Evaluate the baseline model performance:

```bash
python eval-self-edits-baseline.py \
    --experiment_folder=${TTI_DIR}/eval_base_model \
    --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct \
    --lora_checkpoints_folder=${LORA_DIR}/self-edit/eval_RL_iteration_1_8_epoch \
    --temperature=0 \
    --n_sample=1 \
    --data_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json \
    --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format \
    --num_examples=9
```

## SEAL RL Iteration 0

### Create Self-Edits on Eval Set (RL Iteration 1)

Generate self-edits using the first RL iteration model:

```bash
python self-edit.py \
    --experiment_name=eval_RL_iteration_1 \
    --challenge_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json \
    --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json \
    --model_name=${LORA_DIR}/self-edit/training_set_iteration_1/RL_trained_model_iteration_1 \
    --n_tasks=10 \
    --n_self_edits_per_task=5
```

## Notes

- All experiments use the Llama-3.2-1B-Instruct base model
- The experiments are designed to iteratively improve performance through self-editing and reinforcement learning
- Evaluation is performed on filtered ARC-AGI datasets for both training and evaluation sets
- LoRA (Low-Rank Adaptation) is used for efficient fine-tuning with various rank configurations

## Python code-generation mode

Several scripts (`self-edit.py`, `predict_baseline.py`, `predict_custom.py`, `eval-self-edits.py`, and `eval-self-edits-baseline.py`)
accept a `--code_mode` flag that switches the prompt formatting to the
`PythonSolverMessageRepresenter`. In this configuration the model receives ARC grids as nested
Python lists and must respond with executable Python code that defines the entry point below:

```python
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """Solve the ARC task for the provided test input grid.

    Args:
        grid: The test input grid represented as a list of lists of integers.

    Returns:
        A list of lists of integers representing the predicted output grid.
    """
    ...
```

The response must be valid Python source code (helper functions are allowed) and include the
docstring exactly as shown. Only standard-library imports are permitted; the prompt explicitly
allows modules such as `typing`, `collections`, `itertools`, `functools`, `math`, `statistics`,
`heapq`, and `copy`. Downstream parsers expect the function to return a nested list of integers
representing the predicted output grid.

## Iterative self-improvement controller

Use `few-shot/iterative_code_trainer.py` to run a local end-to-end self-improvement loop that
samples candidate programs, executes them with the sandbox, logs replay data, and periodically
fine-tunes a LoRA adapter on successful transcripts. The controller shares the same prompt builder
and executor utilities as the inference scripts, enabling reproducible code-mode experiments.

Example invocation:

```bash
python iterative_code_trainer.py \
    --model /path/to/base-model \
    --tasks data/arc-agi_training_challenges.json \
    --solutions data/arc-agi_training_solutions.json \
    --output-dir runs/iterative-training \
    --tasks-per-iteration 4 \
    --generations-per-task 6 \
    --min-successes-for-update 8
```

Key outputs:

- `replay.jsonl`: JSONL replay buffer with prompts, generated code, execution traces, and rewards.
- `adapters/iter-XXXX/`: Saved LoRA adapters produced whenever the success threshold is met. The
  script automatically loads the most recent adapter into the active inference model.

Adjust the CLI flags to control sampling temperature, new-token limits, LoRA hyper-parameters, and
update cadence to match the desired experimental protocol.
