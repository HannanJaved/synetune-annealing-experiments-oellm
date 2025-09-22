#!/bin/bash
#SBATCH --job-name=test_evals
#SBATCH --output=$LOG_DIR/%a.stdout
#SBATCH --error=$LOG_DIR/%a.stderr
#SBATCH --cpus-per-task=1
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=AIFAC_L01_028
#SBATCH --time=1439
#SBATCH --exclude=lrdn[1711-3456]

# ml Python  # cluster specific
# ml Cuda  # cluster specific
source /leonardo/home/userexternal/hmahadik/myenv/bin/activate
export HF_HOME=/leonardo/home/userexternal/hmahadik/.cache/
export LM_EVAL_OUTPUT_PATH=/leonardo/home/userexternal/hmahadik/evals
# export LM_EVAL_RESULTS_PATH=/leonardo/home/userexternal/hmahadik/logs/synetune-initialruns/run5/eval_results  # Commented out - now passed from sbatch
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # number of GPU specific
    
# LOG_DIR is now passed from sbatch
mkdir -p "$LOG_DIR"
    
export PYTHONPATH=$PYTHONPATH:../eval-synetune

cd .

# checks that at least $SLURM_ARRAY_TASK_ID lines exists in python arguments.
[ "$(wc -l < python-args.txt)" -lt "$SLURM_ARRAY_TASK_ID" ] && { echo "Error: python-args.txt has fewer lines than max_num_line ($max_num_line)."; echo "ERROR"; exit 1; }

bash main_script.sh `sed -n "$(( $SLURM_ARRAY_TASK_ID + 1 ))p" python-args.txt`
