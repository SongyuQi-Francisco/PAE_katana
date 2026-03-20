import json
import os
import argparse
import sys
import math
import glob

# Setup Paths
AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(AGENT_DIR, "data")
PROFILES_PATH = os.path.join(DATA_DIR, "user_profiles.json")

def calculate_hr(hits, total):
    return (hits / total) if total > 0 else 0

def calculate_ndcg(hits_ndcg, total):
    return (hits_ndcg / total) if total > 0 else 0

def analyze_split(task_set, version):
    execution_logs_path = os.path.join(DATA_DIR, "execution_logs.jsonl")
    if not os.path.exists(execution_logs_path):
        print(f"Error: execution_logs.jsonl not found at {execution_logs_path}")
        return

    # Load profiles
    with open(PROFILES_PATH, 'r') as f:
        profiles = json.load(f)

    # Load Ground Truths
    WORKSPACE_ROOT = os.path.dirname(AGENT_DIR)
    GT_DIR = os.path.join(WORKSPACE_ROOT, "AgentSocietyChallenge", "example", "track2", task_set, "groundtruth")
    
    executions = []
    with open(execution_logs_path, 'r') as f:
        for line in f:
            if line.strip():
                executions.append(json.loads(line))

    # Reverse executions to ensure we process the latest if there are duplicates
    executions.reverse()

    gt_map = {}
    gt_files = glob.glob(os.path.join(GT_DIR, "groundtruth_*.json"))
    for gf in gt_files:
        with open(gf, 'r') as f:
            data = json.load(f)
            gt_filename = os.path.basename(gf)
            task_filename = gt_filename.replace("groundtruth_", "task_")
            task_dir = os.path.join(os.path.dirname(os.path.dirname(gf)), "tasks")
            task_file = os.path.join(task_dir, task_filename)
            with open(task_file, 'r') as tf:
                t_data = json.load(tf)
                user_id = t_data.get("user_id")
                gt_map[user_id] = data.get("ground truth")

    stats = {
        "cold_start": {"total": 0, "hr1": 0, "hr3": 0, "hr5": 0, "ndcg1": 0, "ndcg3": 0, "ndcg5": 0},
        "long_tail": {"total": 0, "hr1": 0, "hr3": 0, "hr5": 0, "ndcg1": 0, "ndcg3": 0, "ndcg5": 0}
    }

    processed_users = set()

    for ex in executions:
        uid = ex['user_id']
        if uid in processed_users:
            continue
        
        gt = gt_map.get(uid)
        if not gt: continue
        processed_users.add(uid)

        profile = profiles.get(uid, {})
        is_cold = "[COLD START]" in profile.get("reasoning", "") or not profile
        group = "cold_start" if is_cold else "long_tail"
        
        stats[group]["total"] += 1
        
        preds = ex.get('predicted_top_5', [])
        if not preds and ex.get('predicted_top_1'):
            preds = [ex['predicted_top_1']]

        # Evaluate HR@K and NDCG@K
        for k in [1, 3, 5]:
            preds_k = preds[:k]
            if gt in preds_k:
                stats[group][f"hr{k}"] += 1
                rank = preds_k.index(gt) + 1
                stats[group][f"ndcg{k}"] += 1.0 / math.log2(rank + 1)

    print(f"\\n===== Performance Split Analysis: {task_set} ({version}) =====")
    for group, data in stats.items():
        count = data["total"]
        hr1 = calculate_hr(data["hr1"], count)
        hr3 = calculate_hr(data["hr3"], count)
        hr5 = calculate_hr(data["hr5"], count)
        ndcg1 = calculate_ndcg(data["ndcg1"], count)
        ndcg3 = calculate_ndcg(data["ndcg3"], count)
        ndcg5 = calculate_ndcg(data["ndcg5"], count)
        
        print(f"Group: {group.upper()}")
        print(f"  Total Tasks: {count}")
        print(f"  HR@1: {hr1:.2%} | HR@3: {hr3:.2%} | HR@5: {hr5:.2%}")
        print(f"  NDCG@1: {ndcg1:.4f} | NDCG@3: {ndcg3:.4f} | NDCG@5: {ndcg5:.4f}")
    
    output_path = os.path.join(DATA_DIR, f"split_analysis_{task_set}_{version}.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\\nAnalysis saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_set", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    analyze_split(args.task_set, args.version)
