#!/usr/bin/env python3
"""Full evaluation: 7 tasks, before and after LoRA, using HFLM-based analog model."""

import sys
import os
import json
import time
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import custom_analog_model
import lm_eval

parser = argparse.ArgumentParser()
parser.add_argument('--tasks', type=str, default=None,
                    help='Comma-separated task list (default: all 7)')
parser.add_argument('--limit', type=int, default=None,
                    help='Limit examples per task (for quick testing)')
parser.add_argument('--lora_path', type=str, default="saved_lora_models/default",
                    help='Path to trained LoRA weights')
parser.add_argument('--batch_size', type=int, default=4)
cli_args = parser.parse_args()

ALL_TASKS = ["sciq", "piqa", "arc_easy", "arc_challenge", "boolq", "winogrande", "hellaswag"]
TASKS = [t.strip() for t in cli_args.tasks.split(",")] if cli_args.tasks else ALL_TASKS
BATCH_SIZE = cli_args.batch_size

BASE_MODEL_ARGS = (
    "model_name=unsloth/Meta-Llama-3.1-8B,"
    "use_analog=True,"
    "output_noise_level=0,"
    "weight_noise_level=0.067,"
    "eval_with_noise=True"
)

LORA_PATH = cli_args.lora_path

CONFIGS = [
    ("before_lora", BASE_MODEL_ARGS),
    ("after_lora", BASE_MODEL_ARGS + f",lora_path={LORA_PATH}"),
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

all_results = {}

for config_name, model_args in CONFIGS:
    print("\n" + "=" * 70)
    print(f"CONFIG: {config_name}")
    print("=" * 70)

    config_results = {}
    config_start = time.time()

    for task in TASKS:
        print(f"\n--- {config_name} / {task} ---")
        task_start = time.time()

        try:
            results = lm_eval.simple_evaluate(
                model="custom_analog",
                model_args=model_args,
                tasks=[task],
                num_fewshot=0,
                batch_size=BATCH_SIZE,
                device="cuda:0",
                limit=cli_args.limit,
            )

            task_elapsed = time.time() - task_start
            task_data = results.get("results", {}).get(task, {})
            config_results[task] = task_data

            acc = task_data.get("acc,none", "N/A")
            acc_norm = task_data.get("acc_norm,none", "N/A")
            print(f"  {task}: acc={acc}, acc_norm={acc_norm} ({task_elapsed:.1f}s)")

        except Exception as e:
            task_elapsed = time.time() - task_start
            print(f"  {task}: FAILED ({e}) ({task_elapsed:.1f}s)")
            config_results[task] = {"error": str(e)}

    config_elapsed = time.time() - config_start
    print(f"\n{config_name} completed in {config_elapsed:.1f}s")
    all_results[config_name] = config_results

# Save JSON results
json_path = os.path.join(RESULTS_DIR, "eval_results.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nJSON results saved to {json_path}")

# Print summary table
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Task':<18} {'Before LoRA acc':<18} {'After LoRA acc':<18} {'Before norm':<18} {'After norm':<18}")
print("-" * 90)
for task in TASKS:
    before = all_results.get("before_lora", {}).get(task, {})
    after = all_results.get("after_lora", {}).get(task, {})
    b_acc = before.get("acc,none", "N/A")
    a_acc = after.get("acc,none", "N/A")
    b_norm = before.get("acc_norm,none", "N/A")
    a_norm = after.get("acc_norm,none", "N/A")
    b_acc_s = f"{b_acc:.4f}" if isinstance(b_acc, (int, float)) else str(b_acc)
    a_acc_s = f"{a_acc:.4f}" if isinstance(a_acc, (int, float)) else str(a_acc)
    b_norm_s = f"{b_norm:.4f}" if isinstance(b_norm, (int, float)) else str(b_norm)
    a_norm_s = f"{a_norm:.4f}" if isinstance(a_norm, (int, float)) else str(a_norm)
    print(f"{task:<18} {b_acc_s:<18} {a_acc_s:<18} {b_norm_s:<18} {a_norm_s:<18}")

# Save markdown summary
md_path = os.path.join(RESULTS_DIR, "eval_results_summary.md")
with open(md_path, "w") as f:
    f.write("# Evaluation Results Summary\n\n")
    f.write("## Configuration\n")
    f.write("- **Model**: Meta-Llama-3.1-8B (unsloth)\n")
    f.write("- **Analog**: per-channel noise 0.067, eval_with_noise=True\n")
    f.write("- **LoRA**: r=16, targets=[q,k,v,o,gate,up,down]_proj\n")
    f.write(f"- **LoRA weights**: {LORA_PATH}\n")
    f.write("- **Evaluation**: lm-evaluation-harness, 0-shot, batch_size=4\n\n")
    f.write("## Results\n\n")
    f.write("| Task | Before LoRA (acc) | After LoRA (acc) | Before (acc_norm) | After (acc_norm) |\n")
    f.write("|------|-------------------|------------------|-------------------|------------------|\n")
    for task in TASKS:
        before = all_results.get("before_lora", {}).get(task, {})
        after = all_results.get("after_lora", {}).get(task, {})
        b_acc = before.get("acc,none", "N/A")
        a_acc = after.get("acc,none", "N/A")
        b_norm = before.get("acc_norm,none", "N/A")
        a_norm = after.get("acc_norm,none", "N/A")
        fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        f.write(f"| {task} | {fmt(b_acc)} | {fmt(a_acc)} | {fmt(b_norm)} | {fmt(a_norm)} |\n")

print(f"Markdown summary saved to {md_path}")
print("\nDone!")
