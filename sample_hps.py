import pathlib
import argparse
import pandas
import numpy as np

from syne_tune.config_space import randint
from syne_tune.optimizer.schedulers.searchers.conformal.conformal_quantile_regression_searcher import ConformalQuantileRegression

from generate_pretrain_scripts import generate_scripts



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=2, help="Number of nodes")
    parser.add_argument("--csv-file", type=str, default="~/experiments/moe/megatron_analysis.csv", help="Path to the csv file")
    args = parser.parse_args()


    nodes = args.nodes
    gpu_per_node = 4
    total_num_gpus = nodes * gpu_per_node
    csv_path = pathlib.Path(args.csv_file)

    df = pandas.read_csv(csv_path)

    df = df[df['nodes'] == nodes]

    config_space = {'global_batch_size': randint(8, 512),
                    'tensor_model_parallel_size': randint(1, total_num_gpus),
                    'pipeline_model_parallel_size': randint(1, total_num_gpus),
                    'context_parallel_size': randint(1, total_num_gpus),
                    'expert_model_parallel_size': randint(1, total_num_gpus),
                    }

    cqr = ConformalQuantileRegression(config_space=config_space)


    for iter, row in df.iterrows():
        config = {k: row[k] for k in config_space.keys()}
        metric = row['mean_tflops']
        if np.isnan(metric):
            continue
        print(f"Iteration {iter}: config={config}, metric={metric}")
        cqr.on_trial_complete(iter, config, metric)

    new_config = None
    for i in range(1000):
        candidate = cqr.suggest()

        micro_batch_times_data_parallel_size = df['micro_batch_size'][0] * candidate['tensor_model_parallel_size'] * candidate['pipeline_model_parallel_size'] * candidate['expert_model_parallel_size']
        if candidate['global_batch_size'] % micro_batch_times_data_parallel_size != 0:
            continue
        elif total_num_gpus % (candidate['tensor_model_parallel_size'] * candidate['pipeline_model_parallel_size'] * candidate['expert_model_parallel_size'] * candidate['context_parallel_size'])  != 0:
            continue
        else:
            print(f'sampled {i} configurations')
            new_config = candidate
            break

    if new_config is None:
        print("No valid configuration found after 1000 attempts.")
        return
    print(f"New configuration: {new_config}")
    generate_scripts(nodes, new_config)


if __name__ == '__main__':
    main()