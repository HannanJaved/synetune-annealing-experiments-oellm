import os
import json
import csv

# Base directory containing all runs
base_dir = "/leonardo/home/userexternal/hmahadik/logs/synetune-initialruns"

# Output CSV file
output_csv = "/leonardo_work/AIFAC_L01_028/hmahadik/synetune-experiments/aggregated_eval_results.csv"

# Initialize CSV headers
csv_headers = ["Name", "iteration", "task", "metric", "value"]

# Collect all unique data paths
all_data_paths = set()

# Collect results
results = []

# Iterate through all run directories
for run_dir in sorted(os.listdir(base_dir)):
    run_path = os.path.join(base_dir, run_dir)
    if not os.path.isdir(run_path):
        continue

    # Walk the run directory tree and look for any eval_results directories.
    # For each eval_results found, read the data_paths.txt from its parent directory
    # (where data_paths.txt is now located) and then process the eval results.
    for root, dirs, files in os.walk(run_path):
        if "eval_results" not in dirs:
            continue

        eval_results_dir = os.path.join(root, "eval_results")
        parent_dir = root

        # Parse data_paths.txt from the parent directory of eval_results
        data_paths_file = os.path.join(parent_dir, "data_paths.txt")
        data_list = []
        if os.path.exists(data_paths_file):
            with open(data_paths_file, "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("#") or not line or line.startswith("DATA_PATHS=") or line == "(" or line == ")":
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    # Accept integer or float weights (e.g. 14 or 7.84)
                    try:
                        w = float(parts[0])
                        weight = int(w) if w.is_integer() else w
                        path = " ".join(parts[1:])
                    except ValueError:
                        # If first token is not numeric, treat the whole line as the path
                        weight = None
                        path = line
                    data_list.append((path, weight))
                elif len(parts) == 1:
                    path = parts[0]
                    data_list.append((path, None))

        # Add to all_data_paths
        for path, _ in data_list:
            all_data_paths.add(path)

        # Iterate through iteration directories inside this eval_results
        for iter_dir in sorted(os.listdir(eval_results_dir)):
            iter_path = os.path.join(eval_results_dir, iter_dir)
            if not os.path.isdir(iter_path):
                continue

            # Iterate through task directories
            for task_dir in sorted(os.listdir(iter_path)):
                task_path = os.path.join(iter_path, task_dir)
                if not os.path.isdir(task_path):
                    continue

                # Look for the results JSON file
                for file in os.listdir(task_path):
                    if file.endswith(".json"):
                        json_path = os.path.join(task_path, file)
                        with open(json_path, "r") as f:
                            data = json.load(f)

                        # Create weight dict
                        weight_dict = {path: weight for path, weight in data_list}

                        # Extract results for each task
                        if task_dir in ["mmlu", "mmlu_continuation"]:
                            results_dict = data.get("results", {})
                            if results_dict:
                                first_task = next(iter(results_dict))
                                metrics = results_dict[first_task]
                                for metric, value in metrics.items():
                                    if metric in ["acc,none", "acc_norm,none"]:
                                        row = [run_dir, str(int(iter_dir.split("_")[-1])), first_task, metric, value] + [weight_dict.get(path, None) for path in sorted(all_data_paths)]
                                        results.append(row)
                        else:
                            for task, metrics in data.get("results", {}).items():
                                for metric, value in metrics.items():
                                    if metric in ["acc,none", "acc_norm,none"]:
                                        row = [run_dir, str(int(iter_dir.split("_")[-1])), task, metric, value] + [weight_dict.get(path, None) for path in sorted(all_data_paths)]
                                        results.append(row)

# Update headers with data paths
csv_headers += sorted(list(all_data_paths))

# Write results to CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_headers)  # Write headers
    writer.writerows(results)    # Write data

print(f"Results aggregated into {output_csv}")
