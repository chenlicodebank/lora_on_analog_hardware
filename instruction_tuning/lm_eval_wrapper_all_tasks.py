#!/usr/bin/env python3
"""
Wrapper script for lm_eval with wandb integration.
Evaluates 12 tasks individually and logs results to wandb.
"""

import sys
import os
import argparse
import subprocess
import json
import wandb

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our custom model to register it
try:
    import custom_analog_model
    print("✓ Successfully imported and registered custom_analog model")
except Exception as e:
    print(f"✗ Failed to import custom_analog_model: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model on 12 tasks with wandb logging')
    parser.add_argument('--run_name', type=str, required=True, help='Name for the wandb run')
    parser.add_argument('--model', type=str, default='custom_analog', help='Model to evaluate')
    parser.add_argument('--model_args', type=str, required=True, help='Model arguments')
    parser.add_argument('--num_fewshot', type=int, default=0, help='Number of few-shot examples')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--batch_size', type=str, default='auto:4', help='Batch size')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for results')
    parser.add_argument('--trust_remote_code', action='store_true', help='Trust remote code')
    parser.add_argument('--show_config', action='store_true', help='Show configuration')
    return parser.parse_args()

def init_wandb(run_name):
    """Initialize wandb with the specified run name"""
    try:
        wandb.login()
        wandb.init(
            project="lm_eval_tasks",
            name=run_name,
            reinit=True
        )
        print(f"✓ Initialized wandb run: {run_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize wandb: {e}")
        return False

def run_single_task(task, args):
    """Run evaluation for a single task"""
    print(f"\n{'='*60}")
    print(f"EVALUATING TASK: {task}")
    print(f"{'='*60}")
    
    # Create output directory for this task
    task_output_path = f"{args.output_path}_{task}"
    os.makedirs(task_output_path, exist_ok=True)
    
    # Build command
    cmd = [
        'python', 'lm_eval_wrapper.py',
        '--model', args.model,
        '--model_args', args.model_args,
        '--tasks', task,
        '--num_fewshot', str(args.num_fewshot),
        '--device', args.device,
        '--batch_size', args.batch_size,
        '--output_path', task_output_path
    ]
    
    if args.trust_remote_code:
        cmd.append('--trust_remote_code')
    if args.show_config:
        cmd.append('--show_config')
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the evaluation
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✓ Successfully completed evaluation for {task}")
            
            # Try to parse results from the output
            try:
                # Look for results.json file in the output directory
                results_file = os.path.join(task_output_path, 'results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results_data = json.load(f)
                    
                    # Extract metrics for this task
                    task_results = results_data.get('results', {}).get(task, {})
                    
                    # Log to wandb
                    log_data = {}
                    for metric, value in task_results.items():
                        if isinstance(value, (int, float)):
                            log_data[f"{task}/{metric}"] = value
                    
                    if log_data:
                        wandb.log(log_data)
                        print(f"✓ Logged {len(log_data)} metrics to wandb for {task}")
                    else:
                        print(f"⚠ No numeric metrics found for {task}")
                
                else:
                    print(f"⚠ Results file not found for {task}: {results_file}")
                    
            except Exception as e:
                print(f"⚠ Failed to parse/log results for {task}: {e}")
            
            return True
            
        else:
            print(f"✗ Failed to evaluate {task}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout while evaluating {task}")
        return False
    except Exception as e:
        print(f"✗ Exception while evaluating {task}: {e}")
        return False

def main():
    args = parse_args()
    
    # Define the 10 tasks to evaluate
    tasks = [
        'piqa',
        'hellaswag',
        'winogrande',
        'arc_easy',
        'arc_challenge',
        'boolq',
        'sciq',
        'copa',
        'openbookqa'
    ]
    
    print(f"Starting evaluation of {len(tasks)} tasks with wandb logging")
    print(f"Run name: {args.run_name}")
    print(f"Tasks: {', '.join(tasks)}")
    
    # Initialize wandb
    if not init_wandb(args.run_name):
        print("Failed to initialize wandb, exiting...")
        sys.exit(1)
    
    # Log configuration
    wandb.config.update({
        "model": args.model,
        "model_args": args.model_args,
        "num_fewshot": args.num_fewshot,
        "device": args.device,
        "batch_size": args.batch_size,
        "tasks": tasks
    })
    
    # Run evaluation for each task
    successful_tasks = 0
    failed_tasks = []
    
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] Processing task: {task}")
        
        if run_single_task(task, args):
            successful_tasks += 1
        else:
            failed_tasks.append(task)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {len(failed_tasks)}")
    if failed_tasks:
        print(f"Failed tasks: {', '.join(failed_tasks)}")
    
    # Log summary to wandb
    wandb.log({
        "summary/total_tasks": len(tasks),
        "summary/successful_tasks": successful_tasks,
        "summary/failed_tasks": len(failed_tasks),
        "summary/success_rate": successful_tasks / len(tasks)
    })
    
    # Finish wandb run
    wandb.finish()
    
    print(f"✓ Evaluation completed. Results logged to wandb run: {args.run_name}")

if __name__ == "__main__":
    main() 