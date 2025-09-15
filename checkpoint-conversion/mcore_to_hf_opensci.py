# Copyright (c) 2023 Alibaba PAI, Nvidia Megatron-LM Team and Taishi Nakamura.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
import sys
from collections.abc import Mapping, Sequence
import logging

import numpy as np
import torch
from huggingface_hub import save_torch_state_dict
from transformers import AutoConfig, AutoTokenizer


@torch.inference_mode()
def clone_state_dict(elem):
    """clone all tensors in the elem to cpu device."""
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        elem = elem.clone()
    elif isinstance(elem, (np.ndarray, str)):
        pass
    elif isinstance(elem, Mapping):
        elem = dict(elem)
        for k, v in elem.items():
            elem[k] = clone_state_dict(v)
        elem = elem_type(elem)
    elif isinstance(elem, Sequence):
        elem = list(elem)
        for i in range(len(elem)):
            elem[i] = clone_state_dict(elem[i])
        elem = elem_type(elem)
    return elem


def add_args(parser):
    parser.add_argument(
        "--megatron-path",
        type=str,
        required=True,
        help="Base directory of Megatron repository",
    )

    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        required=True,
        help="Path to the directory containing checkpoints to convert.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Path to the directory to save converted checkpoints. If not provided, it is inferred from --load_dir.",
    )

    parser.add_argument("--world_size", type=int, default=1, help=("world_size"))

    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--source_model",
        type=str,
        default=None,
        help=("Path to the pretrained model or model identifier from huggingface.co"),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument("--print-checkpoint-structure", action="store_true")

    return parser


megatron_to_transformers = {"self_attention.linear_proj": "self_attn.o_proj"}

tensor_parallel_params_mg = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.linear_proj.weight",
    "self_attention.linear_qkv.weight",
    "self_attention.linear_proj.bias",
    "self_attention.linear_qkv.bias",
]

column_split_tensor_parallel_params_mg = ["self_attention.linear_proj"]


def get_checkpoint_sub_dir_name(tp_rank, pp_rank, pp_size):
    sub_dir_name = f"mp_rank_{tp_rank:02d}"
    if pp_size > 1:
        sub_dir_name = f"{sub_dir_name}_{pp_rank:03d}"
    return sub_dir_name


def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.
    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = [{"model": {}} for i in range(tp_size)]
    for tp_index, i in enumerate(range(tp_size)):
        sub_dir_name = get_checkpoint_sub_dir_name(i, pp_rank, pp_size)
        logging.info(f"Loading {sub_dir_name}...")
        # Since distrib_optim.pt is unnecessary, explicitly specify model_optim_rng.pt instead.
        checkpoint_path = os.path.join(
            args.load_path, sub_dir_name, "model_optim_rng.pt"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Could not find model_optim_rng.pt in {os.path.join(args.load_path, sub_dir_name)}. "
                f"Available files: {os.listdir(os.path.join(args.load_path, sub_dir_name))}"
            )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        tp_state_dicts[tp_index]["model"].update(state_dict["model"])
    return tp_state_dicts


def megatron_to_transformers_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :] for compatibility with later versions
    of NVIDIA Megatron-LM. The inverse operation is performed inside Megatron-LM to read checkpoints:
    https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209 If param is the weight tensor of the
    self-attention block, the returned tensor will have to be transposed one more time to be read by HuggingFace GPT2.
    This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def transformers_to_megatron_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to the one compatible with respective NVIDIA Megatron-LM chekpoint versions. Input
    is [num_splits * num_heads * hidden_size, :] and output is [num_heads * hidden_size * num_splits, :] for version
    1.0 and [num_heads * num_splits * hidden_size, :] for version 2.0 and later. If param is the weight tensor of the
    self-attention block, the param needs to be already transposed before calling this function.
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # Input is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            logging.info(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        logging.info(f"{msg}: {val.size()}")
    else:
        logging.info(f"{msg}: {val}")


def get_element_from_dict_by_path(d, path):
    if path not in d:
        d[path] = {}
    d = d[path]
    return d


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Search in directory above this
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    )
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    # Load Megatron-LM checkpoint arguments from the state dict
    possible_state_paths: list[str] = [os.path.join(args.load_path)]
    logging.info(f"DEBUG: possible_state_paths: {possible_state_paths}")
    state_path = None
    for p in possible_state_paths:
        if os.path.exists(p):
            state_path = p
            logging.info(f"Loading Megatron-LM checkpoint arguments from: {state_path}")
            break
    assert state_path is not None, f"Cannot find state path in {possible_state_paths}"
    possible_sub_dirs = [
        "mp_rank_00",
        "mp_rank_00_000",
        "mp_rank_00_dp_000",
        "mp_rank_00_000_dp_000",
    ]

    state_dirs = os.listdir(state_path)
    for sub_dir in possible_sub_dirs:
        if sub_dir in state_dirs:
            rank0_checkpoint_path = os.path.join(
                state_path, sub_dir, "model_optim_rng.pt"
            )
            break
    logging.info(
        f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}"
    )  # type: ignore
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")  # type: ignore
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint instead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, local_files_only=True)

    config = AutoConfig.from_pretrained(args.source_model, local_files_only=True, trust_remote_code=True)
    # Update config with Megatron args
    config.architectures = [args.architecture]
    config.attention_bias = megatron_args.add_qkv_bias
    config.attention_dropout = megatron_args.attention_dropout
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.hidden_act = "silu"  # swiglu
    config.hidden_size = megatron_args.hidden_size
    config.initializer_range = megatron_args.init_method_std
    config.intermediate_size = megatron_args.ffn_hidden_size
    config.layer_norm_eps = megatron_args.norm_epsilon
    config.max_position_embeddings = megatron_args.seq_length
    config.mlp_bias = megatron_args.add_bias_linear
    config.model_type = args.model_type
    config.num_attention_heads = megatron_args.num_attention_heads
    config.num_hidden_layers = megatron_args.num_layers
    config.num_key_value_heads = (
        args.num_key_value_heads
        if args.num_key_value_heads is not None
        else megatron_args.num_query_groups
    )
    config.qk_layernorm = megatron_args.qk_layernorm
    config.rms_norm_eps = megatron_args.norm_epsilon
    config.rope_scaling = (
        None
        if megatron_args.use_rope_scaling is False
        else megatron_args.use_rope_scaling
    )
    config.rope_theta = megatron_args.rotary_base
    config.tie_word_embeddings = not megatron_args.untie_embeddings_and_output_weights

    output_state_dict = {}

    # checkpoint_version = state_dict.get("checkpoint_version", 3.0)
    tp_size = args.target_tensor_model_parallel_size
    pp_size = args.target_pipeline_model_parallel_size

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Convert.
    logging.info("Converting")

    # Embeddings
    logging.info("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)

    logging.info("tp_state_dicts", tp_state_dicts[0]["model"].keys())

    # import pdb
    # pdb.set_trace()

    # Convert and store the word embeddings.
    word_embeddings = []

    # import pdb
    # pdb.set_trace()
    embeddings = tp_state_dicts[0]["model"]["embedding.word_embeddings.weight"]
    for tp_rank in range(tp_size):
        embeddings = tp_state_dicts[tp_rank]["model"][
            "embedding.word_embeddings.weight"
        ]
        word_embeddings.append(embeddings)

    word_embeddings = torch.cat(word_embeddings, dim=0)
    word_embeddings = word_embeddings.to(dtype)
    output_state_dict["model.embed_tokens.weight"] = word_embeddings.clone()
    # Reset the vocab size
    config.vocab_size = word_embeddings.shape[0]

    # Transformer Layers
    logging.info("Converting transformer layers")
    # The number of heads.
    heads = config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers // pp_size

    hidden_size = config.hidden_size
    num_groups = config.num_key_value_heads

    for pp_rank in range(pp_size):
        if pp_size > 0:
            logging.info(f"Converting pipeline parallel rank {pp_rank}")
            tp_state_dicts = get_megatron_sharded_states(
                args, tp_size, pp_size, pp_rank
            )

        # The transformer.

        path = "model"

        # Extract the layers.
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            if key.endswith("_extra_state"):
                continue
            if "layer_norm_weight" not in key and "linear_fc" in key:
                key_list = key.split(".")
                layer_id = int(key_list[2]) + pp_rank * num_layers

                if "weight" in key:
                    dim = 1 if "linear_fc2" in key else 0
                    params = torch.cat(
                        [val]
                        + [
                            get_element_from_dict_by_path(
                                tp_state_dicts[tp_rank], f"{path}"
                            )[key]
                            for tp_rank in range(1, tp_size)
                        ],
                        dim=dim,
                    ).to(dtype)

                    if "linear_fc2" in key:
                        output_state_dict[
                            f"model.layers.{layer_id}.mlp.down_proj.weight"
                        ] = params
                    else:
                        params_split = [
                            torch.chunk(i, 2, 0)
                            for i in torch.chunk(params, tp_size, 0)
                        ]
                        output_state_dict[
                            f"model.layers.{layer_id}.mlp.gate_proj.weight"
                        ] = torch.cat([i[0] for i in params_split])
                        output_state_dict[
                            f"model.layers.{layer_id}.mlp.up_proj.weight"
                        ] = torch.cat([i[1] for i in params_split])
                elif "bias" in key:
                    params = torch.cat(
                        [val]
                        + [
                            get_element_from_dict_by_path(
                                tp_state_dicts[tp_rank], f"{path}"
                            )[key]
                            for tp_rank in range(1, tp_size)
                        ],
                        dim=0,  # bias is always combined with dim=0
                    ).to(dtype)

                    if "linear_fc2" in key:
                        output_state_dict[
                            f"model.layers.{layer_id}.mlp.down_proj.bias"
                        ] = params
                    else:
                        params_split = [
                            torch.chunk(i, 2, 0)
                            for i in torch.chunk(params, tp_size, 0)
                        ]
                        output_state_dict[
                            f"model.layers.{layer_id}.mlp.gate_proj.bias"
                        ] = torch.cat([i[0] for i in params_split])
                        output_state_dict[
                            f"model.layers.{layer_id}.mlp.up_proj.bias"
                        ] = torch.cat([i[1] for i in params_split])
                continue

            new_key = key.replace("decoder.", "")
            if "layer_norm_weight" in new_key:
                new_key += ".weight"
            # Match the name.
            m = layer_re.match(new_key)
            # Stop if that's not a layer
            if m is None:
                continue

            # The index of the layer.
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)

            # The name of the layer.
            layer_name = f"model.layers.{layer_idx}"

            # logging.info(layer_name, op_name, weight_or_bias)

            if op_name + "." + weight_or_bias not in tensor_parallel_params_mg:
                params = val.to(dtype)
            else:
                if weight_or_bias == "weight":
                    dim = 1 if op_name in column_split_tensor_parallel_params_mg else 0
                else:  # bias
                    dim = 0

                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(
                            tp_state_dicts[tp_rank], f"{path}"
                        )[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            if "q_layernorm" in op_name:
                output_state_dict[
                    layer_name + ".self_attn.q_layernorm." + weight_or_bias
                ] = params.clone()
                continue
            elif "k_layernorm" in op_name:
                output_state_dict[
                    layer_name + ".self_attn.k_layernorm." + weight_or_bias
                ] = params.clone()
                continue
            # For layernorm(s), simply store the layer norm.
            elif op_name.endswith("layer_norm_weight") or op_name.endswith("layernorm"):
                if "qkv" in op_name:
                    output_state_dict[
                        layer_name + "." + "input_layernorm" + "." + weight_or_bias
                    ] = params.clone()
                elif "mlp.linear_fc1" in op_name:
                    output_state_dict[
                        layer_name
                        + "."
                        + "post_attention_layernorm"
                        + "."
                        + weight_or_bias
                    ] = params.clone()
                continue

            # Transpose the QKV matrix.
            elif (
                op_name == "attention.linear_qkv"
                or op_name == "self_attention.linear_qkv"
            ) and weight_or_bias == "weight":
                logging.info(
                    f"num_groups: {num_groups}, hidden_size_per_head: {hidden_size_per_head}"
                )
                logging.info(f"op_name: {op_name}, weight_or_bias: {weight_or_bias}")

                all_qkvs = [
                    i.reshape(
                        num_groups // args.target_tensor_model_parallel_size,
                        (
                            heads // num_groups * hidden_size_per_head
                            + 2 * hidden_size_per_head
                        ),
                        hidden_size,
                    )
                    for i in torch.chunk(
                        params, args.target_tensor_model_parallel_size, 0
                    )
                ]
                split_size = heads // num_groups * hidden_size_per_head
                all_qs = torch.cat(
                    [i[:, :split_size, :].reshape(-1, hidden_size) for i in all_qkvs]
                )
                all_kvs = torch.cat(
                    [i[:, split_size:, :].reshape(-1, hidden_size) for i in all_qkvs]
                )

                checkpoint_version = 3.0
                out_q = megatron_to_transformers_fix_query_key_value_ordering(
                    all_qs, checkpoint_version, 1, heads, hidden_size_per_head
                )

                out_kv = megatron_to_transformers_fix_query_key_value_ordering(
                    all_kvs, checkpoint_version, 2, num_groups, hidden_size_per_head
                )
                out_kv = torch.chunk(out_kv, 2)

                output_state_dict[layer_name + ".self_attn.q_proj.weight"] = (
                    out_q.clone()
                )
                output_state_dict[layer_name + ".self_attn.k_proj.weight"] = out_kv[
                    0
                ].clone()
                output_state_dict[layer_name + ".self_attn.v_proj.weight"] = out_kv[
                    1
                ].clone()

            elif (
                op_name == "attention.linear_qkv"
                or op_name == "self_attention.linear_qkv"
            ) and weight_or_bias == "bias":
                logging.info("num_groups", num_groups)
                logging.info("hidden_size_per_head", hidden_size_per_head)
                logging.info("op_name", op_name)
                logging.info("weight_or_bias", weight_or_bias)

                all_qkv_biases = [
                    i.reshape(
                        num_groups // args.target_tensor_model_parallel_size,
                        (
                            heads // num_groups * hidden_size_per_head
                            + 2 * hidden_size_per_head
                        ),
                    )
                    for i in torch.chunk(
                        params, args.target_tensor_model_parallel_size, 0
                    )
                ]

                split_size = heads // num_groups * hidden_size_per_head
                all_q_biases = torch.cat(
                    [i[:, :split_size].reshape(-1) for i in all_qkv_biases]
                )
                all_kv_biases = torch.cat(
                    [i[:, split_size:].reshape(-1) for i in all_qkv_biases]
                )

                checkpoint_version = 3.0
                out_q_bias = megatron_to_transformers_fix_query_key_value_ordering(
                    all_q_biases.unsqueeze(-1),
                    checkpoint_version,
                    1,
                    heads,
                    hidden_size_per_head,
                ).squeeze(-1)

                out_kv_bias = megatron_to_transformers_fix_query_key_value_ordering(
                    all_kv_biases.unsqueeze(-1),
                    checkpoint_version,
                    2,
                    num_groups,
                    hidden_size_per_head,
                ).squeeze(-1)
                out_kv_bias = torch.chunk(out_kv_bias, 2)

                output_state_dict[layer_name + ".self_attn.q_proj.bias"] = (
                    out_q_bias.clone()
                )
                output_state_dict[layer_name + ".self_attn.k_proj.bias"] = out_kv_bias[
                    0
                ].clone()
                output_state_dict[layer_name + ".self_attn.v_proj.bias"] = out_kv_bias[
                    1
                ].clone()

            # Transpose the weights.
            elif weight_or_bias == "weight":
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + "." + out_name + "." + "weight"] = (
                    params.clone()
                )
            # Handle biases
            elif weight_or_bias == "bias":
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + "." + out_name + "." + "bias"] = (
                    params.clone()
                )

    if config.num_hidden_layers != (layer_idx + 1):
        raise ValueError(
            f"Expected {config.num_hidden_layers} layers but found {layer_idx + 1}"
        )

    # The final layernorm.
    logging.info("Converting final layernorm")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    try:
        output_state_dict["model.norm.weight"] = (
            params["decoder.final_layernorm.weight"].to(dtype).clone()
        )
    except Exception as e:
        logging.debug(f"Error converting final layernorm: {e}")
        output_state_dict["model.norm.weight"] = (
            params["decoder.final_norm.weight"].to(dtype).clone()
        )

    # For LM head, transformers' wants the matrix to weight embeddings.
    logging.info("Converting LM head")
    if not config.tie_word_embeddings:
        # If we're not tying weights.
        params = torch.cat(
            [
                get_element_from_dict_by_path(
                    tp_state_dicts[i]["model"], "output_layer.weight"
                )
                for i in range(tp_size)
            ]
        )
        output_state_dict["lm_head.weight"] = params.to(dtype).clone()

    # It should be done!
    logging.info("Conversion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    logging.info("Saving checkpoint...")
    config.save_pretrained(args.save_path)
    save_torch_state_dict(
        state_dict=output_state_dict,
        save_directory=args.save_path,
        safe_serialization=True,
    )
    logging.info(f"Model weights saved in {args.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_key_value_heads",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="OpensciForCausalLM",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="opensci",
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="EleutherAI/gpt-neox-20b",
    )
    parser = add_args(parser)
    args = parser.parse_args()

    print(f"Arguments: {args}")

    if args.convert_checkpoint_from_megatron_to_transformers:
        print("Converting from Megatron-LM to Transformers format...")
        if not os.path.isdir(args.load_dir):
            logging.error(f"Load directory {args.load_dir} does not exist.")
            return

        if args.save_dir is None:
            if "torch" in args.load_dir:
                args.save_dir = args.load_dir.replace("torch", "hf")
            else:
                args.save_dir = os.path.join(
                    os.path.dirname(args.load_dir.rstrip("/")),
                    os.path.basename(args.load_dir.rstrip("/")) + "_hf",
                )
            logging.info(f"Save directory not provided. Using: {args.save_dir}")

        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Created save directory: {args.save_dir}")

        # Convert the single iteration directly (args.load_dir points to iter_XXXXXX)
        if os.path.basename(args.load_dir).startswith("iter_"):
            logging.info(f"Converting single iteration: {args.load_dir}")
            iter_args = argparse.Namespace(**vars(args))
            iter_args.load_path = args.load_dir
            iter_args.save_path = args.save_dir

            try:
                convert_checkpoint_from_megatron_to_transformers(iter_args)
            except Exception as e:
                logging.error(
                    f"Failed to convert checkpoint in {iter_args.load_path}: {e}"
                )
        else:
            # Handle multiple iterations (if load_dir contains multiple iter_ directories)
            for dirname in sorted(os.listdir(args.load_dir)):
                if dirname.startswith("iter_") and os.path.isdir(
                    os.path.join(args.load_dir, dirname)
                ):
                    logging.info(f"Found checkpoint directory: {dirname}")
                    iter_args = argparse.Namespace(**vars(args))
                    iter_args.load_path = os.path.join(args.load_dir, dirname)
                    iter_args.save_path = os.path.join(args.save_dir, dirname)

                    if os.path.exists(iter_args.save_path):
                        logging.warning(
                            f"Save path {iter_args.save_path} already exists. Skipping conversion."
                        )
                        continue

                    try:
                        convert_checkpoint_from_megatron_to_transformers(iter_args)
                    except Exception as e:
                        logging.error(
                            f"Failed to convert checkpoint in {iter_args.load_path}: {e}"
                        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
