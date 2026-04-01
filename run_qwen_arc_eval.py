#!/usr/bin/env python3
"""Evaluate Qwen v7 LoRA on ARC-AGI-2 evaluation tasks.

Loads the fine-tuned LoRA adapter, generates code solutions for
each ARC task, executes them, and scores against ground truth.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1 (GPU 0 = Theory Radar)

import json
import time
import logging
import traceback
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()


def load_model():
    """Load Qwen 2.5-7B with v7 LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model = "Qwen/Qwen2.5-7B"
    adapter_path = "/home/claude/arc-agi-2/models/qwen2.5-7b-arc-code-lora"

    log.info("Loading base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    log.info("Loading LoRA adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    log.info("Model loaded on GPU 1")
    return model, tokenizer


def format_grid(grid):
    """Format a 2D grid as a string."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def format_task_prompt(task):
    """Format an ARC task as a prompt for the code-generation model."""
    prompt = "# ARC-AGI Task\n# Given input-output training pairs, write a Python function that transforms the input grid to the output grid.\n\n"

    for i, pair in enumerate(task["train"]):
        prompt += f"# Training example {i+1}:\n"
        prompt += f"# Input:\n# {format_grid(pair['input']).replace(chr(10), chr(10) + '# ')}\n"
        prompt += f"# Output:\n# {format_grid(pair['output']).replace(chr(10), chr(10) + '# ')}\n\n"

    prompt += "# Test input:\n"
    prompt += f"# {format_grid(task['test'][0]['input']).replace(chr(10), chr(10) + '# ')}\n\n"
    prompt += "# Write a Python function `solve(grid)` that takes a 2D list of integers and returns the transformed 2D list.\n"
    prompt += "def solve(grid):\n"

    return prompt


def generate_solution(model, tokenizer, prompt, max_new_tokens=512):
    """Generate a code solution from the prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated


def execute_solution(code_str, test_input):
    """Execute the generated code on the test input. Returns grid or None."""
    try:
        # Prepend the function definition
        full_code = "def solve(grid):\n" + code_str

        namespace = {}
        exec(full_code, namespace)

        if "solve" not in namespace:
            return None

        result = namespace["solve"]([row[:] for row in test_input])

        # Validate: must be a list of lists of ints
        if not isinstance(result, list):
            return None
        for row in result:
            if not isinstance(row, list):
                return None
            for val in row:
                if not isinstance(val, int):
                    return None

        return result
    except Exception:
        return None


def grids_equal(a, b):
    """Check if two grids are identical."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        if ra != rb:
            return False
    return True


def main():
    log.info("ARC-AGI-2 Evaluation — Qwen v7 LoRA")

    # Load data
    with open("/home/claude/arc-agi-2/data/arc-agi_evaluation_challenges.json") as f:
        challenges = json.load(f)
    with open("/home/claude/arc-agi-2/data/arc-agi_evaluation_solutions.json") as f:
        solutions = json.load(f)

    log.info("Loaded %d evaluation tasks", len(challenges))

    # Load model
    model, tokenizer = load_model()

    # Evaluate
    correct = 0
    total = 0
    errors = 0
    results = {}
    t0 = time.time()

    for task_id, task in challenges.items():
        total += 1

        try:
            prompt = format_task_prompt(task)
            code = generate_solution(model, tokenizer, prompt)

            test_input = task["test"][0]["input"]
            predicted = execute_solution(code, test_input)

            if predicted is None:
                errors += 1
                results[task_id] = {"status": "exec_error", "code": code[:200]}
                continue

            expected = solutions[task_id][0]

            if grids_equal(predicted, expected):
                correct += 1
                results[task_id] = {"status": "correct"}
                log.info("  [%d/%d] %s: CORRECT", total, len(challenges), task_id)
            else:
                results[task_id] = {"status": "wrong"}

        except Exception as e:
            errors += 1
            results[task_id] = {"status": "error", "msg": str(e)[:100]}

        if total % 10 == 0:
            elapsed = time.time() - t0
            log.info("  Progress: %d/%d correct=%d errors=%d (%.1fs/task)",
                     total, len(challenges), correct, errors, elapsed / total)

    elapsed = time.time() - t0
    accuracy = correct / max(total, 1)

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS: %d/%d correct (%.1f%%)", correct, total, 100 * accuracy)
    log.info("Errors: %d, Time: %.0fs (%.1fs/task)", errors, elapsed, elapsed / max(total, 1))
    log.info("=" * 60)

    # Save results
    with open("qwen_v7_eval_results.json", "w") as f:
        json.dump({
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "errors": errors,
            "time_seconds": elapsed,
            "per_task": results,
        }, f, indent=2)

    log.info("Saved qwen_v7_eval_results.json")


if __name__ == "__main__":
    main()
