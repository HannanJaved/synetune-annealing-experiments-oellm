# filepath: /leonardo_work/AIFAC_L01_028/hmahadik/synetune-experiments/sample_data_weights.py
import pathlib
import argparse
import pandas as pd
import numpy as np

from syne_tune.config_space import randint, uniform
from syne_tune.optimizer.schedulers.searchers.conformal.conformal_quantile_regression_searcher import ConformalQuantileRegression

# Define groups and their paths
DATA_GROUPS = {
    'nemotron': [
        '/leonardo_work/AIFAC_L01_028/datasets/Nemotron-cc-2024/tokenized/quality-high_kind-actual/GPT-NeoX/merged_0',
        '/leonardo_work/AIFAC_L01_028/datasets/Nemotron-cc-2024/tokenized/quality-high_kind-actual/GPT-NeoX/merged_1',
        '/leonardo_work/AIFAC_L01_028/datasets/Nemotron-cc-2024/tokenized/quality-high_kind-synthetic/GPT-NeoX/merged_0',
        '/leonardo_work/AIFAC_L01_028/datasets/Nemotron-cc-2024/tokenized/quality-high_kind-synthetic/GPT-NeoX/merged_1',
        '/leonardo_work/AIFAC_L01_028/datasets/Nemotron-cc-2024/tokenized/quality-high_kind-synthetic/GPT-NeoX/merged_2',
        '/leonardo_work/AIFAC_L01_028/datasets/Nemotron-cc-2024/tokenized/quality-high_kind-synthetic/GPT-NeoX/merged_3',
    ],
    'finemath': [
        '/leonardo_work/AIFAC_L01_028/datasets/tokenized/HuggingFaceTB-FineMath/GPT-NeoX/finemath-4plus/merged',
        '/leonardo_work/AIFAC_L01_028/datasets/tokenized/HuggingFaceTB-FineMath/GPT-NeoX/infiwebmath-3plus/merged',
    ],
    'starcoder': [
        '/leonardo_work/AIFAC_L01_028/datasets/tokenized/bigcode-starcoderdata/GPT-NeoX/merged',
    ],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", type=str, default="/leonardo_work/AIFAC_L01_028/hmahadik/synetune-experiments/aggregated_eval_results.csv", help="Path to the aggregated CSV file")
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv_file)
    df = pd.read_csv(csv_path)

    # Filter to acc,none and acc_norm,none metrics
    df = df[df['metric'].isin(['acc,none', 'acc_norm,none'])]

    # Group by Name (run) and compute average value as metric
    run_metrics = df.groupby('Name')['value'].mean().reset_index()
    run_metrics.rename(columns={'value': 'avg_acc'}, inplace=True)

    # Get weight columns (data paths)
    weight_cols = [col for col in df.columns if col.startswith('/leonardo')]

    # Merge back to get weights per run
    run_data = df[['Name'] + weight_cols].drop_duplicates()

    # Calculate total number of data paths for equal split assumption
    total_paths = sum(len(paths) for paths in DATA_GROUPS.values())
    equal_per_path = 100 / total_paths

    # Config space for group weights (raw, will be normalized to sum to 100)
    config_space = {
        'nemotron_weight': uniform(0.1, 10),
        'finemath_weight': uniform(0.1, 10),
        'starcoder_weight': uniform(0.1, 10),
    }
    # config_space = {
    #     'nemotron_weight': randint(1, 10),
    #     'finemath_weight': randint(1, 10),
    #     'starcoder_weight': randint(1, 10),
    # }

    cqr = ConformalQuantileRegression(config_space=config_space)

    # Train on past runs
    for _, row in run_metrics.iterrows():
        run_name = row['Name']
        metric = row['avg_acc']
        if np.isnan(metric):
            continue

        # Get weights for this run
        run_row = run_data[run_data['Name'] == run_name]
        if run_row.empty:
            continue

        # Extract raw group weights
        raw_config = {}
        for group, paths in DATA_GROUPS.items():
            group_weights = [run_row[path].values[0] for path in paths if path in weight_cols]
            if group_weights and not pd.isna(group_weights[0]):
                raw_config[f'{group}_weight'] = float(group_weights[0])
            else:
                # Missing, assume equal split
                raw_config[f'{group}_weight'] = equal_per_path * len(paths)

        # Normalize to sum to 100
        total_sum = sum(raw_config.values())
        config = {k: (v / total_sum) * 100 for k, v in raw_config.items()}
        print(f"Training on {run_name}: config={config}, metric={metric}")
        cqr.on_trial_complete(run_name, config, metric)

    # Sample new config
    new_config = None

    # for i in range(1000):
    #     candidate = cqr.suggest()
    #     # No additional constraints, accept any
    #     new_config = candidate
    #     print(f'Sampled {i} configurations')
    #     break

    candidate = cqr.suggest()
    new_config = candidate # No additional constraints, accept any
        
    if new_config is None:
        print("No valid configuration found.")
        return

    # Normalize new config to sum to 100
    total_sum = sum(new_config.values())
    new_config = {k: (v / total_sum) * 100 for k, v in new_config.items()}

    print(f"New configuration (normalized): {new_config}")

    # Generate full weights dict
    full_weights = {}
    for group, paths in DATA_GROUPS.items():
        group_weight = new_config[f'{group}_weight']
        path_weight = group_weight / len(paths)
        for path in paths:
            full_weights[path] = path_weight

    print("New data weights (sum to 100):")
    for path, weight in full_weights.items():
        print(f"{path}: {weight:.2f}")

if __name__ == '__main__':
    main()