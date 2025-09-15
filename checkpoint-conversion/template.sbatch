#!/bin/bash -x
#SBATCH --account={account}
#SBATCH --partition={partition} 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --qos=qos_dbg
#SBATCH --output={convert_logs_dir}/%x_%j.out
#SBATCH --error={convert_logs_dir}/%x_%j.err
#SBATCH --job-name=convert_full


IMAGE={container_image}

APPTAINER_ARGS="singularity exec --nv \
                --bind $HOME:$HOME \
                --bind $WORK:$WORK \
                --bind $SCRATCH:$SCRATCH \
                ${IMAGE}"

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

MEGATRON_OPEN_SCI_PATH="{opensci_megatron_path}"
export PYTHONPATH=$MEGATRON_OPEN_SCI_PATH:$PYTHONPATH

OPEN_SCI_HF_PATH={opensci_hf_path}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS_PER_NODE=4
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

echo NUM_NODES=$NUM_NODES
echo NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
echo NUM_GPUS=$NUM_GPUS

TARGET_TP_SIZE=1
TARGET_PP_SIZE=1
WORLD_SIZE=$((TARGET_TP_SIZE * TARGET_PP_SIZE))

TRAIN_LOGS_PATH={train_logs_path}

CHECKPOINTS_DIR="{save_checkpoints_dir}"
TORCH_CKPT_PATH="{torch_ckpt_path}"
SAVE_CHECKPOINT_PATH="${CHECKPOINTS_DIR}/hf"

echo CHECKPOINTS_DIR=$CHECKPOINTS_DIR
echo TORCH_CKPT_PATH=$TORCH_CKPT_PATH
echo SAVE_CHECKPOINT_PATH=$SAVE_CHECKPOINT_PATH

ITERATION_TO_CONVERT="{iteration_to_convert}"
LATEST_ITER=$ITERATION_TO_CONVERT

echo "LATEST_ITER: $LATEST_ITER"
SAVE_CHECKPOINT_PATH="${SAVE_CHECKPOINT_PATH}/iter_${LATEST_ITER}"
TORCH_CKPT_ITER_PATH="${TORCH_CKPT_PATH}/iter_${LATEST_ITER}"

mkdir -p ${SAVE_CHECKPOINT_PATH}

# Execute pre-run command inside the container if specified
if [ ! -z "{pre_run_cmd}" ]; then
    ${APPTAINER_ARGS} bash -c "{pre_run_cmd}"
fi

# Convert the model to HF
source {venv_path}/bin/activate || exit 1

# cd "$MEGATRON_OPEN_SCI_PATH"

intermediate_size={ffn_hidden_size}
max_position_embeddings={max_seq_length}
NUM_KEY_VALUE_HEADS={num_attn_heads}

cat <<EOF > ${SAVE_CHECKPOINT_PATH}/config.json
{
    "_name_or_path": "",
    "architectures": [
      "OpensciForCausalLM"
    ],
    "attention_bias": true,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_opensci.OpensciConfig",
        "AutoModel": "modeling_opensci.OpensciPreTrainedModel",
        "AutoModelForCausalLM": "modeling_opensci.OpensciForCausalLM"
      },
    "bos_token_id": 0,
    "eos_token_id": 0,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": $intermediate_size,
    "max_position_embeddings": $max_position_embeddings,
    "mlp_bias": true,
    "model_type": "opensci",
    "num_attention_heads": 32,
    "num_hidden_layers": 24,
    "num_key_value_heads": $NUM_KEY_VALUE_HEADS,
    "pretraining_tp": 1,
    "qk_layernorm": true,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "rope_theta": 10000,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.48.3",
    "use_cache": true,
    "vocab_size": 50304
  }
EOF

cp -r $OPEN_SCI_HF_PATH/sample/modeling_opensci.py "${SAVE_CHECKPOINT_PATH}"
cp -r $OPEN_SCI_HF_PATH/sample/configuration_opensci.py "${SAVE_CHECKPOINT_PATH}"

# Tokenizer
cp -r $OPEN_SCI_HF_PATH/sample/tokenizer* "${SAVE_CHECKPOINT_PATH}"
cp -r $OPEN_SCI_HF_PATH/sample/special_tokens_map.json "${SAVE_CHECKPOINT_PATH}"
cp -r $OPEN_SCI_HF_PATH/sample/vocab.json "${SAVE_CHECKPOINT_PATH}"

cd {current_dir}

$APPTAINER_ARGS \
  python -u mcore_to_hf_opensci.py \
    --megatron-path "${MEGATRON_OPEN_SCI_PATH}" \
    --load_dir "${TORCH_CKPT_ITER_PATH}" \
    --save_dir "${SAVE_CHECKPOINT_PATH}" \
    --source_model "${OPEN_SCI_HF_PATH}/sample" \
    --target_tensor_model_parallel_size ${TARGET_TP_SIZE} \
    --target_pipeline_model_parallel_size ${TARGET_PP_SIZE} \
    --target_params_dtype "bf16" \
    --world_size ${WORLD_SIZE} \
    --convert_checkpoint_from_megatron_to_transformers \
    --num_key_value_heads ${NUM_KEY_VALUE_HEADS}

