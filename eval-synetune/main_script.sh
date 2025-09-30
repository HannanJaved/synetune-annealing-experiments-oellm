#!/bin/bash

set -e

# Script to evaluate a huggingface model or a list of models included in a path
# Takes TASKS, NUM_FEW_SHOT and MODEL_PATH_OR_NAME as arguments.
if [ $# -lt 3 ]; then
    echo "Error: Not enough arguments provided."
    echo "Usage: $0 TASKS NUM_FEWSHOT MODEL_PATH_OR_NAME"
    exit 1
fi

TASKS=$1
NUM_FEWSHOT=$2

# can be a path containing "model.safetensors" files or a model name from huggingface
MODEL_PATH_OR_NAME=$3

export WANDB_MODE=offline
BATCH_SIZE='auto'

echo "BATCH_SIZE: $BATCH_SIZE"
echo "LM_EVAL_OUTPUT_PATH: $LM_EVAL_OUTPUT_PATH"

# avoid issue "sqlite3.OperationalError: database is locked"
export OUTLINES_CACHE_DIR=/tmp/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID/$MODEL_PATH_OR_NAME

# LM_EVAL_RESULTS_PATH="${LM_EVAL_RESULTS_PATH}" 
LM_EVAL_RESULTS_PATH="${LM_EVAL_RESULTS_PATH:-/leonardo/home/userexternal/hmahadik/logs/synetune-initialruns/run18/eval_results}"

mkdir -p $LM_EVAL_OUTPUT_PATH

OUTPUT_PATH=$LM_EVAL_OUTPUT_PATH/$SLURM_ARRAY_JOB_ID/
TASK_STR=${TASKS//,/_}

# Helper to extract iter_xxxx from model path
extract_iter() {
    local path="$1"
    if [[ "$path" =~ (iter_[0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "unknown_iter"
    fi
}

if [ -d "$MODEL_PATH_OR_NAME" ]; then
  # we evaluate all models found recursively in MODEL_PATH_OR_NAME
  echo "Evaluating all models found recursively in folder $MODEL_PATH_OR_NAME with $NUM_FEW_SHOT few-shots on the following tasks $TASKS."
  # gets all files "model.safetensors" defined in the current path and launch evaluation for each of them
  mapfile -t SAFETENSOR_FILES < <(find "$MODEL_PATH_OR_NAME" -type f -name "model.safetensors")

  # Check if any files were found
  if [ ${#SAFETENSOR_FILES[@]} -eq 0 ]; then
      echo "No model.safetensors files found in $MODEL_PATH_OR_NAME."
      exit 0
  fi

  echo "Found ${#SAFETENSOR_FILES[@]} model.safetensors files"

  # Loop through all model files found recursively and evaluate all of them
  for FILE_PATH in "${SAFETENSOR_FILES[@]}"; do
    MODEL_PATH=$(dirname "$FILE_PATH")
    MODEL_STR=`echo $MODEL_PATH | sed 's#/leonardo_work/EUHPC_E03_068/tcarsten/converted_checkpoints/open-sci-ref_model-##'`
    WANDB_NAME="$SLURM_ARRAY_JOB_ID-$MODEL_STR"

    ITER_DIR=$(extract_iter "$MODEL_PATH")
    TASK_STR=${TASKS//,/_}
    RESULT_DIR="$LM_EVAL_RESULTS_PATH/$ITER_DIR/$TASK_STR"
    mkdir -p "$RESULT_DIR"

    echo "Evaluate $MODEL_PATH"
    echo "results will be stored in $RESULT_DIR/results.json"
    accelerate launch -m lm_eval --model hf \
        --model_args pretrained=$MODEL_PATH,trust_remote_code=True\
        --tasks $TASKS \
        --output_path $RESULT_DIR/results.json \
        --batch_size $BATCH_SIZE \
        --num_fewshot $NUM_FEWSHOT \
        --trust_remote_code \
        --wandb_args project=lm-eval-harness-integration,name=$WANDB_NAME
  done
else
  # we evaluate the single model passed as argument
  echo "Evaluate model from huggingface $MODEL_PATH_OR_NAME on $TASKS with $NUM_FEW_SHOT fewshots."
  # avoid having strings that are too long, we could also pick the last part of the string
  MODEL_STR=`echo $MODEL_PATH_OR_NAME | sed 's#/leonardo_work/EUHPC_E03_068/tcarsten/converted_checkpoints/open-sci-ref_model-##'`
  WANDB_NAME="$SLURM_ARRAY_JOB_ID-$MODEL_STR"

  ITER_DIR=$(extract_iter "$MODEL_PATH_OR_NAME")
  TASK_STR=${TASKS//,/_}
  RESULT_DIR="$LM_EVAL_RESULTS_PATH/$ITER_DIR/$TASK_STR"
  mkdir -p "$RESULT_DIR"

  accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH_OR_NAME,trust_remote_code=True\
    --tasks $TASKS \
    --output_path $RESULT_DIR/results.json \
    --batch_size $BATCH_SIZE \
    --num_fewshot $NUM_FEWSHOT \
    --trust_remote_code \
    --wandb_args project=lm-eval-harness-integration,name=$WANDB_NAME
fi

# python llmeval/reorganize_results.py --output-dir evals/results --evals-dir evals

