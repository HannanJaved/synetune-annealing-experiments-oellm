#!/bin/bash

#SBATCH --job-name=Nemotron-Synth
#SBATCH --output=/leonardo/home/userexternal/hmahadik/logs/synetune-initialruns/run13/actual_run.out
#SBATCH --error=/leonardo/home/userexternal/hmahadik/logs/synetune-initialruns/run13/actual_run.err
#SBATCH --time=08:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --account=AIFAC_L01_028
#SBATCH --partition=boost_usr_prod
#SBATCH --threads-per-core=1
#SBATCH --qos=boost_qos_lprod
#SBATCH --exclude=lrdn[1711-3456]

# For quick tests on max 2 nodes: #SBATCH --qos=boost_qos_dbg or for more #SBATCH --qos=boost_qos_lprod
# For large node numbers (> 64 nodes): add #SBATCH --qos=boost_qos_bprod
# 
# time #SBATCH --time=4-00:00:00
# Megatron-LM Slurm Script for Leonardo HPC
# Based on scripts from Sampo Pyysalo, Jenia Jitsev, Joerg Franke, Marianna Nezhurina
# WARNING: ADAPT corresponding paths carefully

# Usage:
#     sbatch annealing.sh

######################################################################
# ENVIRONMENT SETUP AND GENERAL CONFIGURATION
######################################################################

echo "START $SLURM_JOBID: $(date)"

# Bash "strict mode" (see http://redsymbol.net/articles/unofficial-bash-strict-mode/)
set -euo pipefail

echo "SLURM_JOB_ID: $SLURM_JOB_ID"

CHECKPOINT_PATH="${BASE_PATH}/run13/checkpoints"
TENSORBOARD_DIR="${BASE_PATH}/run13/tensorboard"
mkdir -p "$CHECKPOINT_PATH"
mkdir -p "$TENSORBOARD_DIR"

export BASE_PATH

# Read DATA_PATHS from already created file
DATA_PATHS=($(cat ${BASE_PATH}/run13/data_paths.txt))

# WEIGHTS & BIASES CONFIG
USE_WANDB=1
WANDB_PROJECT="annealing"
WANDB_EXP_NAME="synetune-initial-run13"
WANDB_DIR="${BASE_PATH}/wandb"
export WANDB_MODE="offline"

# HF CONFIG
export HF_HUB_OFFLINE=1

# Path to Megatron-LM repo
MEGATRON_PATH="/leonardo_work/AIFAC_L01_028/hmahadik/Megatron-LM"

# MEGATRON CACHE
MEGATRON_CACHE_BASE=$SCRATCH
MEGATRON_CACHE_FOLDER="${MEGATRON_CACHE_BASE}/${USER}"
export MEGATRON_CACHE="${MEGATRON_CACHE_FOLDER}/MEGATRON_CACHEDIR"
mkdir -p "$MEGATRON_CACHE_FOLDER"
mkdir -p "$MEGATRON_CACHE"

# APPTAINER CACHE
export APPTAINER_CACHEDIR="${MEGATRON_CACHE_FOLDER}/APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${MEGATRON_CACHE_FOLDER}/APPTAINER_TMPDIR"
mkdir -p $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR

# Container image
CONTAINER="/leonardo_work/AIFAC_L01_028/hmahadik/container_pretraining_megatron.sif"

# Directories to map into container
BIND_DIRS="${MEGATRON_PATH},${WORK},${FAST},${MEGATRON_CACHE_FOLDER},${HF_HOME}"

# DISTRIBUTED SETUP
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # master node hostname
MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"  # IP numeric address
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT=12345
echo "MASTER_ADDR:MASTER_PORT set to: ${MASTER_ADDR}:${MASTER_PORT}"

# NCCL settings to improve distributed training stability (handling flipping links, irresponsive nodes, etc)
# waiting for 120s in case nodes become irresponsive giving a chance to recover
export NCCL_IB_TIMEOUT=120

export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs
export CUDA_DEVICE_MAX_CONNECTIONS=1


######################################################################
# DATA, MODEL AND PRETRAINING CONFIGURATION
######################################################################

# DATA
# DATA_PATHS=(
#     # NEMOTRON HQ DATA SPLIT 24k
#     14 /leonardo_work/AIFAC_L01_028/datasets/Nemotron-cc-2024/tokenized/quality-high_kind-synthetic/GPT-NeoX/merged_0
#     14 /leonardo_work/AIFAC_L01_028/datasets/Nemotron-cc-2024/tokenized/quality-high_kind-synthetic/GPT-NeoX/merged_1
#     14 /leonardo_work/AIFAC_L01_028/datasets/Nemotron-cc-2024/tokenized/quality-high_kind-synthetic/GPT-NeoX/merged_2
#     14 /leonardo_work/AIFAC_L01_028/datasets/Nemotron-cc-2024/tokenized/quality-high_kind-synthetic/GPT-NeoX/merged_3
#     14 /leonardo_work/AIFAC_L01_028/datasets/Nemotron-cc-2024/tokenized/quality-high_kind-actual/GPT-NeoX/merged_0
#     14 /leonardo_work/AIFAC_L01_028/datasets/Nemotron-cc-2024/tokenized/quality-high_kind-actual/GPT-NeoX/merged_1

#     # CODE - STARCODER DATA
#     6 /leonardo_work/AIFAC_L01_028/datasets/tokenized/bigcode-starcoderdata/GPT-NeoX/merged

#     # MATH - FineWeb Math Data
#     5 /leonardo_work/AIFAC_L01_028/datasets/tokenized/HuggingFaceTB-FineMath/GPT-NeoX/infiwebmath-3plus/merged
#     5 /leonardo_work/AIFAC_L01_028/datasets/tokenized/HuggingFaceTB-FineMath/GPT-NeoX/finemath-4plus/merged
# )    

# # {{DATA_PATHS_PLACEHOLDER}}

SPLIT=949,50,1
VOCAB_FILE="/leonardo_work/AIFAC_L01_028/hmahadik/gpt-neox-20b/vocab.json"
MERGE_FILE="/leonardo_work/AIFAC_L01_028/hmahadik/gpt-neox-20b/merges.txt"
DATA_NUM_WORKERS=4

# MODEL
NUM_LAYERS=22
HIDDEN_SIZE=512
FFN_HIDDEN_SIZE=2256
NUM_ATTENTION_HEADS=8
NUM_QUERY_GROUPS=8    # No GQA when NUM_QUERY_GROUPS=NUM_ATTENTION_HEADS
TIE_WORD_EMBEDDINGS=1
INIT_METHOD_STD=0.02
MAX_POSITION_EMBEDDINGS=4096
SEQ_LENGTH=4096
ROTARY_BASE=10000
ROTARY_PERCENT=1.0
NORM_EPSILON=1e-5

# PARALLELISM
PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=1
PROFILE=0

# OPTIMIZER
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-8
LR=3e-4
MIN_LR=0
WARMUP_FRACTION=0
COOLDOWN_FRACTION=1
CLIP_GRAD=1.0
WEIGHT_DECAY=0.05

# TRAINING
FSDP=0
GLOBAL_BATCH_SIZE=96
# GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=8
RECOMPUTATION=0
# TRAIN_TOKENS=30_000_000_000 # 30B tokens
TRAIN_TOKENS=10_000_000_000 # 10B tokens

# SAVING AND EVALUATION
LOG_INTERVAL=1
SAVE_INTERVAL=2000
EVAL_INTERVAL=5000
EVAL_ITERS=100

# Set CHECKPOINT_FORMAT
# consider adding --async-save with torch_dist
CHECKPOINT_FORMAT="torch"
if (( TENSOR_MODEL_PARALLEL_SIZE > 1 || PIPELINE_MODEL_PARALLEL_SIZE > 1 )); then
    CHECKPOINT_FORMAT="torch_dist"
fi

######################################################################
#
# DERIVED CONFIGURATION SETTINGS
#
# The following settings are derived from the configuration above.
# Do set these directly, as they will be overwritten here.
#
######################################################################

# Check that variables are not set (sanity)
confirm_unset() {
    local varname="$1"
    if [ -n "${!varname+x}" ]; then
	echo "Error: variable '$varname' should not be set." >&2
	exit 1
    fi
}
confirm_unset "TRAIN_ITERS"
confirm_unset "LR_DECAY_ITERS"
confirm_unset "LR_WSD_DECAY_ITERS"

divide_rounding_up() {
    echo $((($1+$2-1)/$2))
}

# Calculate TRAIN_ITERS from TRAIN_TOKENS
TRAIN_TOKENS=${TRAIN_TOKENS//_}    # drop "_" for bash math
ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TRAIN_ITERS=$(divide_rounding_up $TRAIN_TOKENS $ITER_TOKENS)

# Set LR_WARMUP_ITERS and LR_WSD_DECAY_ITERS based on WARMUP_FRACTION
# and COOLDOWN_FRACTION
LR_WARMUP_ITERS=$((TRAIN_ITERS*${WARMUP_FRACTION}))
LR_WSD_DECAY_ITERS=$((TRAIN_ITERS*${COOLDOWN_FRACTION}))

# LR_DECAY_ITERS is simply set to TRAIN_ITERS
LR_DECAY_ITERS=$TRAIN_ITERS

######################################################################
#
# BUILDING COMMAND-LINE ARGUMENTS
#
# The following builds the command-line arguments for
# Megatron-LM/pretrain_gpt.py based on the variables defined above
# (and optionally in any config given to the script). Note that some
# arguments that are not expected to vary are hard-coded here.
#
######################################################################

DATA_ARGS=(
    --data-path "${DATA_PATHS[@]}"
    --data-cache-path ${MEGATRON_CACHE}
    --vocab-file ${VOCAB_FILE}
    --merge-file ${MERGE_FILE}
    --num-workers ${DATA_NUM_WORKERS}
    --split ${SPLIT}
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "EleutherAI/gpt-neox-20b"
    --reset-position-ids
)

MODEL_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --num-attention-heads ${NUM_ATTENTION_HEADS}
    --seq-length ${SEQ_LENGTH}
    --max-position-embeddings ${SEQ_LENGTH}
)

if [ "$NUM_QUERY_GROUPS" != "$NUM_ATTENTION_HEADS" ]; then
    MODEL_ARGS+=(
        --group-query-attention
        --num-query-groups $NUM_QUERY_GROUPS
    )
fi

if [ "$TIE_WORD_EMBEDDINGS" = "0" ]; then
    MODEL_ARGS+=(
	--untie-embeddings-and-output-weights
    )
fi

if [ "$FSDP" = "1" ]; then
    PARALLEL_ARGS=(
	--use-torch-fsdp2
    )
else
    PARALLEL_ARGS=(
	--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
	--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
	--context-parallel-size $CONTEXT_PARALLEL_SIZE
	# --sequence-parallel
	--use-distributed-optimizer
    )
fi

if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS=(
	--use-pytorch-profiler
	--profile-ranks 0
	--profile-step-start 5
	--profile-step-end 7
    )
else
    PROFILE_ARGS=()
fi

MODEL_ARGS+=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --use-flash-attn
    --attention-softmax-in-fp32
    --max-position-embeddings $SEQ_LENGTH
    --seq-length $SEQ_LENGTH
    --position-embedding-type rope
    --rotary-base $ROTARY_BASE
    --disable-bias-linear
    --init-method-std $INIT_METHOD_STD
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon ${NORM_EPSILON}
    --qk-layernorm
    --bf16
    --swiglu
    --distributed-backend nccl
    --distributed-timeout-minutes 10
    # These args substantially improve TFLOP/s/GPU (1.7B model on 54 nodes, 160 vs 140 with vs without)
    --overlap-param-gather
    --overlap-grad-reduce
    # --overlap-param-gather-with-optimizer-step with interleaved pipeline parallelism
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --adam-beta1 $ADAM_BETA1
    --adam-beta2 $ADAM_BETA2
    --adam-eps $ADAM_EPS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style "WSD"
    --lr-wsd-decay-style "linear"
    --lr-warmup-iters $LR_WARMUP_ITERS
    --lr-decay-iters $LR_DECAY_ITERS
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
    --clip-grad $CLIP_GRAD
    --weight-decay $WEIGHT_DECAY
)

# mkdir -p $CHECKPOINT_PATH/torch

OUTPUT_ARGS=(
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITERS
    --tensorboard-dir "$TENSORBOARD_DIR"
    --tensorboard-queue-size 5
    --log-throughput
    --log-progress
#    --async-save
    --log-interval $LOG_INTERVAL
    --ckpt-format "$CHECKPOINT_FORMAT"
    --load "$CHECKPOINT_PATH/torch"
    --save "$CHECKPOINT_PATH/torch"
    --save-interval $SAVE_INTERVAL
)

OUTPUT_ARGS+=(
  --wandb-project "$WANDB_PROJECT"
  --wandb-exp-name "$WANDB_EXP_NAME"
  --wandb-save-dir "$WANDB_DIR"
    )

# Interleaved pipeline scheduling is only possible with pipeline
# parallel degree > 1.
if [ $PIPELINE_MODEL_PARALLEL_SIZE -gt 1 ] && [ $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE -gt 1 ]; then
    PARALLEL_ARGS+=(
	--num-layers-per-virtual-pipeline-stage $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE
    )
fi

if [ "$RECOMPUTATION" = "1" ]; then
    MODEL_ARGS+=(
	--recompute-activations
	--recompute-granularity selective
    )
fi

######################################################################
# RUN
######################################################################

CMD="${MEGATRON_PATH}/pretrain_gpt.py \
    ${DATA_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${PROFILE_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${OUTPUT_ARGS[@]}
    "

echo '=============== CMD: ==============='
echo $CMD
echo '===================================='


LAUNCHER="PYTHONPATH=/mnt/packages/transformers singularity exec \
    --nv \
    --bind $BIND_DIRS \
    --env HF_HUB_OFFLINE=$HF_HUB_OFFLINE \
    --env HF_HOME=$HF_HOME \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_MODE=$WANDB_MODE \
    $CONTAINER \
    python -u -m torch.distributed.run \
      --nproc-per-node $SLURM_GPUS_PER_NODE \
      --nnodes $SLURM_NNODES \
      --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
      --rdzv_backend static \
      --max_restarts 0 \
      --tee 3 \
    "

cd $MEGATRON_PATH

srun \
    --wait=60 \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --threads-per-core=1 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD"

echo "END $SLURM_JOBID: $(date)"

echo "==============TRAINING FINISHED, STARTING CONVERSION===================="

VENV_PATH=/leonardo/home/userexternal/hmahadik/myenv
LOG_PATH=/leonardo/home/userexternal/hmahadik/logs/synetune-initialruns/run13/
MEGATRON_OPENSCI=/leonardo_work/AIFAC_L01_028/hmahadik/Megatron-LM-Open-Sci
HF_OPENSCI=/leonardo_work/AIFAC_L01_028/hmahadik/Open-Sci-hf

echo "==============LAUNCHING CONVERSION JOBS===================="
# Execute the python script and capture its output

OUTPUT=$(srun --nodes=1 --ntasks=1 bash -c "
source $VENV_PATH/bin/activate
python /leonardo_work/AIFAC_L01_028/hmahadik/annealing/checkpoint-conversion/consolidated_conversion_workflow.py \
 --checkpoint_dir $CHECKPOINT_PATH/torch \
 --save_checkpoints_dir $CHECKPOINT_PATH \
 --opensci_hf_path $HF_OPENSCI \
 --opensci_megatron_path $MEGATRON_OPENSCI \
 --convert_logs_dir $LOG_PATH/conversion-logs/ \
 --container_image $CONTAINER \
 --venv_path $VENV_PATH \
 --log_path $LOG_PATH/actual_run.out
")

echo "Python script output:"
echo "${OUTPUT}"

# Extract the job IDs from the specific line we printed
JOB_IDS=$(echo "${OUTPUT}" | grep 'JOB_IDS_TO_WAIT_ON' | cut -d':' -f2-)

if [ -z "$JOB_IDS" ]; then
    echo "Could not find any job IDs to wait for. Exiting."
    exit 1
fi

echo "==============CONVERSION JOBS LAUNCHED===================="
echo "Will start evaluation after jobs finish: ${JOB_IDS}"

echo "==============CONVERSION FINISHED===================="

echo "==============STARTING EVALUATION===================="
LM_EVAL_RESULTS_PATH=$LOG_PATH/eval_results

echo "Submitting the evaluation preparation job with dependency..."
# export HF_CHECKPOINTS_PATH to the sbatch job so the downstream script can use it
sbatch --dependency=afterok:${JOB_IDS} --export=HF_CHECKPOINTS_PATH="${CHECKPOINT_PATH}/hf" /leonardo_work/AIFAC_L01_028/hmahadik/synetune-experiments/prepare_and_launch_eval.sh

echo "Master script has finished submitting all stages."

echo "==============EVALUATION JOBS SUBMITTED===================="