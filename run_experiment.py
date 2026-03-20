import os
import sys
import json
import argparse
import logging
import glob
from datetime import datetime

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Setup Paths - PAE_katana structure
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = AGENT_DIR  # In PAE_katana, root is the same as agent dir

sys.path.insert(0, AGENT_DIR)  # For src.personal_rec_agent and websocietysimulator

from websocietysimulator import Simulator
from websocietysimulator.llm import OpenAILLM
from src.single_stage_rec_agent import SingleStageRecAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key from environment or .env file
def load_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    # Try loading from .env file
    env_path = os.path.join(AGENT_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.strip().split("=", 1)[1]
    raise ValueError("OPENAI_API_KEY not found. Set it in environment or .env file.")

PROPOSAL_API_KEY = load_api_key()


def count_available_tasks(task_dir):
    task_files = sorted(glob.glob(os.path.join(task_dir, "task_*.json")))
    if task_files:
        return len(task_files)
    return len(sorted(glob.glob(os.path.join(task_dir, "*.json"))))


def normalize_groundtruth_dir(source_gt_dir, normalized_root, task_set, version):
    """
    Normalize custom groundtruth files to ASC evaluator format:
    {"ground truth": "<item_id>"}.
    """
    os.makedirs(normalized_root, exist_ok=True)
    target_dir = os.path.join(normalized_root, f"{task_set}_{version}")
    os.makedirs(target_dir, exist_ok=True)

    source_files = sorted(glob.glob(os.path.join(source_gt_dir, "*.json")))
    if not source_files:
        return source_gt_dir

    needs_normalization = False
    for source_file in source_files[:5]:
        with open(source_file, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if "ground truth" not in payload and "ground_truth_id" in payload:
            needs_normalization = True
            break

    if not needs_normalization:
        return source_gt_dir

    for source_file in source_files:
        with open(source_file, "r", encoding="utf-8") as file:
            payload = json.load(file)

        normalized_payload = dict(payload)
        if "ground truth" not in normalized_payload and "ground_truth_id" in normalized_payload:
            normalized_payload["ground truth"] = normalized_payload["ground_truth_id"]

        target_file = os.path.join(target_dir, os.path.basename(source_file))
        with open(target_file, "w", encoding="utf-8") as file:
            json.dump(normalized_payload, file, indent=2, ensure_ascii=False)

    logger.info("Normalized custom groundtruth from %s to %s", source_gt_dir, target_dir)
    return target_dir

def generate_failure_logs(tasks, groundtruths, execution_logs_path, output_failures_path):
    """
    Match execution logs with ground truth to find and save failure cases.
    """
    if not os.path.exists(execution_logs_path):
        logger.warning(f"No execution logs found at {execution_logs_path}")
        return
        
    executions = []
    with open(execution_logs_path, 'r') as f:
        for line in f:
            if line.strip():
                executions.append(json.loads(line))
                
    exec_map = {ex['user_id']: ex for ex in executions}
    
    failures = []
    
    for task, gt in zip(tasks, groundtruths):
        user_id = task.to_dict().get('user_id')
        gt_item = gt.get('ground truth') or gt.get('ground_truth_id')
        
        ex = exec_map.get(user_id)
        if ex:
            pred = ex.get('predicted_top_1')
            if pred != gt_item:
                failures.append({
                    "user_id": user_id,
                    "domain": ex.get('domain'),
                    "used_skill": ex.get('used_skill'),
                    "predicted_top_1": pred,
                    "ground_truth_item": gt_item,
                    "raw_response": ex.get('raw_response')
                })
                
    with open(output_failures_path, 'w', encoding='utf-8') as f:
        json.dump(failures, f, indent=4)
        
    logger.info(f"Identified {len(failures)} total failures.")
    logger.info(f"Failure logs saved to {output_failures_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_set", default="amazon", choices=["amazon", "goodreads", "yelp"],
                        help="Platform to run (used when task_dir not specified)")
    parser.add_argument("--task_dir", type=str, default=None,
                        help="Custom task directory (overrides task_set)")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Custom groundtruth directory (required if task_dir is specified)")
    parser.add_argument("--num_tasks", type=int, default=None,
                        help="Number of tasks to evaluate; default uses all tasks in task_dir")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--version", default="v0", help="Experiment version (e.g., v0, v1)")
    parser.add_argument("--method_name", default="proposal", help="Name of the method (e.g., proposal, baseline_cot)")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation and only run simulation (for training)")
    args = parser.parse_args()

    # Paths - PAE_katana structure
    dataset_dir = os.path.join(AGENT_DIR, "dataset")

    # Use custom directories if provided, otherwise use track2
    if args.task_dir:
        if not args.gt_dir:
            raise ValueError("--gt_dir is required when --task_dir is specified")
        tasks_dir = args.task_dir
        gt_dir = args.gt_dir
        # Infer task_set from directory name for logging
        inferred_task_set = os.path.basename(os.path.dirname(tasks_dir))
        logger.info(f"Using custom task directory: {tasks_dir}")
    else:
        # PAE_katana structure: tasks/track2/{domain}/
        track2_dir = os.path.join(AGENT_DIR, "tasks", "track2")
        tasks_dir = os.path.join(track2_dir, args.task_set, "tasks")
        gt_dir = os.path.join(track2_dir, args.task_set, "groundtruth")
        inferred_task_set = args.task_set
    
    data_dir = os.path.join(AGENT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    if args.task_dir:
        gt_dir = normalize_groundtruth_dir(
            source_gt_dir=gt_dir,
            normalized_root=os.path.join(data_dir, "normalized_groundtruth"),
            task_set=inferred_task_set,
            version=args.version,
        )

    available_tasks = count_available_tasks(tasks_dir)
    if available_tasks == 0:
        raise FileNotFoundError(f"No task files found in {tasks_dir}")

    if args.num_tasks is None or args.num_tasks <= 0:
        effective_num_tasks = available_tasks
    else:
        effective_num_tasks = min(args.num_tasks, available_tasks)
    
    # Root results directory
    global_results_dir = os.path.join(WORKSPACE_ROOT, "results", args.method_name)
    os.makedirs(global_results_dir, exist_ok=True)

    execution_logs_path = os.path.join(data_dir, "execution_logs.jsonl")
    failures_out_path = os.path.join(data_dir, f"failed_cases_{args.version}_{inferred_task_set}.json")
    
    # Clear previous execution logs to avoid mixing runs
    if os.path.exists(execution_logs_path):
        os.remove(execution_logs_path)

    # Tell the agent if it should use evolved skills
    if "evolved" in args.version.lower() or "v1" in args.version.lower():
        os.environ["ENABLE_EVOLUTION"] = "true"
    else:
        os.environ["ENABLE_EVOLUTION"] = "false"

    # Init Simulator
    logger.info(f"Initializing Simulator for {inferred_task_set} task set...")
    os.environ["OPENAI_API_KEY"] = PROPOSAL_API_KEY
    llm = OpenAILLM(api_key=PROPOSAL_API_KEY, model="gpt-4o-mini")
    
    simulator = Simulator(data_dir=dataset_dir, device="cpu", cache=True)
    simulator.set_task_and_groundtruth(task_dir=tasks_dir, groundtruth_dir=gt_dir)
    simulator.set_agent(SingleStageRecAgent)
    simulator.set_llm(llm)
    
    logger.info(f"Running simulation with {effective_num_tasks}/{available_tasks} tasks...")
    simulator.run_simulation(
        number_of_tasks=effective_num_tasks,
        enable_threading=True,
        max_workers=args.max_workers,
    )

    # Generate failure logs for evolution (always needed, even without eval)
    generate_failure_logs(
        tasks=simulator.tasks[:effective_num_tasks],
        groundtruths=simulator.groundtruth_data[:effective_num_tasks],
        execution_logs_path=execution_logs_path,
        output_failures_path=failures_out_path
    )

    # Skip evaluation if --skip_eval flag is set
    if args.skip_eval:
        logger.info("Skipping evaluation (--skip_eval flag set)")
        return

    logger.info("Evaluating results...")
    eval_results = simulator.evaluate()

    completed_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_out_path = os.path.join(
        data_dir,
        f"evaluation_results_{args.version}_{inferred_task_set}_{completed_at}.json",
    )
    global_results_path = os.path.join(
        global_results_dir,
        f"{inferred_task_set}_{args.version}_{completed_at}_metrics.json",
    )

    # Save local results
    with open(results_out_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    # Save to global results folder
    with open(global_results_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    logger.info(f"Metrics saved to local {results_out_path} and global {global_results_path}")


if __name__ == "__main__":
    main()
