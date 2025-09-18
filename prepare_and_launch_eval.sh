#!/bin/bash
#SBATCH --job-name=PrepareEval
#SBATCH --output=/leonardo_work/AIFAC_L01_028/hmahadik/synetune-experiments/eval-synetune/eval_script_logs/prepare_eval.out
#SBATCH --error=/leonardo_work/AIFAC_L01_028/hmahadik/synetune-experiments/eval-synetune/eval_script_logs/prepare_eval.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=AIFAC_L01_028
#SBATCH --partition=boost_usr_prod
#SBATCH --threads-per-core=1
#SBATCH --qos=boost_qos_dbg

echo "✅ All conversion jobs finished. Starting evaluation preparation."

# --- Environment and Paths (Pass these from the master script or define them here) ---
VENV_PATH="/leonardo/home/userexternal/hmahadik/myenv/bin/activate"
TASKS_FILE="/leonardo_work/AIFAC_L01_028/hmahadik/tasks.txt"
EVAL_JOB_DIR="/leonardo_work/AIFAC_L01_028/hmahadik/synetune-experiments/eval-synetune"

source $VENV_PATH

# --- Step 2a: Generate the list of converted model paths ---
# Clear the model_paths.txt file if it exists
> "${EVAL_JOB_DIR}/model_paths.txt"

echo "Scanning for converted models in: $HF_CHECKPOINTS_PATH"
python /leonardo_work/AIFAC_L01_028/hmahadik/print_file_paths.py --directory "$HF_CHECKPOINTS_PATH" --output $EVAL_JOB_DIR

model_paths_file="$EVAL_JOB_DIR/model_paths.txt"

if [ ! -s $model_paths_file ]; then
    echo "❌ Error: No model paths were found after conversion. Exiting."
    exit 1
fi
echo "Found $(wc -l < $model_paths_file) models to evaluate."

# --- Step 2b: Generate python-args.txt ---
echo "Generating python-args.txt..."
# Clear the file if it already exists
> "${EVAL_JOB_DIR}/python-args.txt"

# Loop through each model path
while IFS= read -r model_path; do
  # Loop through each task definition
  while IFS= read -r task_line; do
    # Skip empty lines
    [ -z "$task_line" ] && continue
    
    # Split task_line by semicolon
    tasks=$(echo "$task_line" | cut -d';' -f1 | xargs) # xargs trims whitespace
    num_shot=$(echo "$task_line" | cut -d';' -f2 | xargs)

    # Append the combined line to python-args.txt
    echo "$tasks $num_shot $model_path" >> "${EVAL_JOB_DIR}/python-args.txt"
  done < "$TASKS_FILE"
done < $model_paths_file

echo "✅ python-args.txt successfully created at ${EVAL_JOB_DIR}/python-args.txt"

# --- Step 2c: Submit the Evaluation Job Array ---
cd $EVAL_JOB_DIR

# Get the number of lines to set the array size
ARRAY_SIZE=$(wc -l < ${EVAL_JOB_DIR}/python-args.txt)
if [ "$ARRAY_SIZE" -eq 0 ]; then
    echo "❌ Error: python-args.txt is empty. No jobs to submit."
    exit 1
fi

ARRAY_UPPER_BOUND=$((ARRAY_SIZE - 1))

# Define the max number of concurrent jobs
MAX_CONCURRENT_JOBS=32

echo "Submitting evaluation job array with range 0-${ARRAY_UPPER_BOUND} and max ${MAX_CONCURRENT_JOBS} concurrent tasks."

# Compute log directory
LOG_DIR=$(dirname "$LM_EVAL_RESULTS_PATH")/eval_logs

# Submit the job with the --array flag now on the command line
sbatch --array=0-${ARRAY_UPPER_BOUND}%${MAX_CONCURRENT_JOBS} --export=LM_EVAL_RESULTS_PATH="${LM_EVAL_RESULTS_PATH}",LOG_DIR="${LOG_DIR}" $EVAL_JOB_DIR/slurm_script.sh

echo "🚀 Evaluation job array submitted."
