import argparse
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_iterations_from_checkpoint(
    checkpoint_path: Path,
) -> Tuple[List[str], List[Path]]:
    """
    Determine iterations by scanning the checkpoint directory structure.
    Looks for directories in the format 'iter_0002000', etc.
    """
    iterations = []
    ckpt_iter_paths = []
    # Check if checkpoint_path exists and is a directory
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        logging.warning(
            f"Warning: Checkpoint path {checkpoint_path} does not exist or is not a directory"
        )
        raise ValueError(
            f"Checkpoint path {checkpoint_path} does not exist or is not a directory"
        )

    # Look for iteration directories (format: iter_XXXXXXX)
    for item in checkpoint_path.iterdir():
        if item.is_dir() and item.name.startswith("iter_"):
            iter_num = item.name.split("_")[1]
            if iter_num.isdigit():
                iterations.append(iter_num)
                ckpt_iter_paths.append(item)

    for path in ckpt_iter_paths:
        if os.path.islink(path):
            iter_num = path.name.split("_")[1]
            for item in Path(os.path.realpath(path)).parent.iterdir():
                if (
                    item.is_dir()
                    and item.name.startswith("iter_")
                    and item.name.split("_")[1].isdigit()
                    and int(item.name.split("_")[1]) < int(iter_num)
                ):
                    logging.debug(
                        f"Adding {item} to ckpt_iter_paths for {int(iter_num)} > {int(item.name.split('_')[1])}"
                    )
                    ckpt_iter_paths.append(item)
                    iterations.append(item.name.split("_")[1])

    # Sort iterations numerically and sort ckpt_iter_paths accordingly
    sorted_pairs = sorted(zip(iterations, ckpt_iter_paths), key=lambda x: int(x[0]))
    iterations, ckpt_iter_paths = zip(*sorted_pairs) if sorted_pairs else ([], [])
    iterations = list(iterations)
    ckpt_iter_paths = list(ckpt_iter_paths)

    if not iterations and not ckpt_iter_paths:
        logging.info(f"No iterations found in {checkpoint_path}")
    return iterations, ckpt_iter_paths


def extract_checkpoints_from_path(
    checkpoint_dir: Union[str, Path], log_path: Union[str, Path]
) -> List[Tuple[Path, Path]]:
    """
    Creates a checkpoint and log path tuple from a given checkpoint directory path.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        log_path: Path to the SLURM log file.

    Returns:
        List of tuples containing (checkpoint_path, log_file_path)
    """
    checkpoint_dir = Path(checkpoint_dir)
    log_path = Path(log_path)

    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise ValueError(
            f"Checkpoint directory {checkpoint_dir} does not exist or is not a directory"
        )

    if not log_path.exists() or not log_path.is_file():
        raise ValueError(
            f"SLURM log file {log_path} does not exist or is not a file"
        )

    return [(checkpoint_dir, log_path)]


def extract_model_size(log_path: Path) -> str:
    with open(log_path, "r") as f:
        log_file = f.read()

    if "Total number of parameters in billions" not in log_file:
        raise ValueError("billions not found")

    with open(log_path, "r") as f:
        log_file_lines = f.readlines()

    for line in log_file_lines:
        if "Total number of parameters in billions" in line:
            num = line.strip().split("billions: ")[-1]
            if "1.3" in num:
                return "1.3"
            elif "0.4" in num:
                return "0.4"
            elif "1.7" in num:
                return "1.7"
            elif "0.13" in num:
                return "0.13"


def get_model_config_from_command_line(log_path: Path) -> Optional[Dict[str, int]]:
    """
    Extract model configuration by parsing the arguments block in the log file.

    Args:
        log_path: Path to the log file

    Returns:
        dict: Model configuration parameters or None if not found
    """
    defaults = {
        "0.13": {"FFN_HIDDEN_SIZE": 2256},
        "0.4": {"FFN_HIDDEN_SIZE": 3840},
        "1.3": {"FFN_HIDDEN_SIZE": 5440},
        "1.7": {"FFN_HIDDEN_SIZE": 8192},
    }

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()

        in_args_block = False
        args_lines = []
        for line in lines:
            if " arguments " in line and "------------------------" in line:
                in_args_block = True
                continue
            if " end of arguments " in line and "---------------------" in line:
                break
            if in_args_block:
                args_lines.append(line)

        if not args_lines:
            logging.info(f"No arguments block found in {log_path}")
            return None

        config = {}

        param_map = {
            "num_layers": "NUM_LAYERS",
            "hidden_size": "HIDDEN_SIZE",
            "ffn_hidden_size": "FFN_HIDDEN_SIZE",
            "num_attention_heads": "NUM_ATTN_HEADS",
            "seq_length": "SEQ_LENGTH",
            "max_position_embeddings": "MAX_POSITION_EMBEDDINGS",
        }

        for line in args_lines:
            # e.g. [lrdn1299:0]:  hidden_size ..................................... 512
            line_content = line.split("]:", 1)[-1].strip()
            for param, config_key in param_map.items():
                if line_content.startswith(param):
                    match = re.match(rf"{param}\s*\.+\s*(\S+)", line_content)
                    if match:
                        val_str = match.group(1).strip()
                        try:
                            config[config_key] = int(val_str)
                        except ValueError:
                            logging.warning(
                                f"Could not parse integer value for {param}: {val_str}"
                            )
                        break

        model_size = extract_model_size(log_path=log_path)
        logging.debug(f"model size is {model_size}")

        try:
            for v in param_map.values():
                if v not in config:
                    if (
                        model_size
                        and model_size in defaults
                        and v in defaults.get(model_size, {})
                    ):
                        config[v] = defaults[model_size][v]
                        logging.debug(f"setting {v} as :{defaults[model_size][v]}")
        except Exception as e:
            logging.error(e)

        if not config:
            logging.debug(
                f"No model configuration parameters found in arguments block: {log_path}"
            )
            return None

        return config

    except Exception as e:
        logging.error(f"Error parsing arguments block from log file {log_path}: {e}")
        return None


def get_converted_iterations(save_checkpoints_dir: str) -> List[str]:
    """
    Get the list of iterations that have already been converted for a given model.
    Checks for the existence of model.safetensors in the hf directory to ensure
    conversion was successful.

    Args:
        save_checkpoints_dir: Directory where converted checkpoints are saved

    Returns:
        List of iteration strings that have already been successfully converted
    """
    converted_iterations = []
    model_dir = Path(save_checkpoints_dir)

    if not model_dir.exists():
        return converted_iterations

    # Check hf directory for successfully converted iterations
    hf_dir = model_dir / "hf"
    if hf_dir.exists():
        for item in hf_dir.iterdir():
            if item.is_dir() and item.name.startswith("iter_"):
                iter_num = item.name.split("_")[1]
                if iter_num.isdigit():
                    # Check if model.safetensors exists to verify successful conversion
                    safetensors_file = item / "model.safetensors"
                    if safetensors_file.exists():
                        converted_iterations.append(iter_num)
                        logging.debug(
                            f"Found successfully converted iteration {iter_num}"
                        )
                    else:
                        logging.info(
                            f"Iteration {iter_num} for exists but model.safetensors missing - conversion likely failed"
                        )

    return sorted(converted_iterations, key=int)


def convert_checkpoint_consolidated(
    venv_path: Union[str, Path],
    log_path: Path,
    iterations: List[str],
    ckpt_iter_paths: List[Path],
    checkpoint_dir: Union[str, Path],
    save_checkpoints_dir: str,
    opensci_megatron_path: str,
    opensci_hf_path: str,
    convert_logs_dir: str,
    account: str,
    partition: str,
    container_image: str,
    tokenizer_cache_dir: str,
    model_config: Optional[Dict[str, int]] = None,
) -> None:
    """
    Convert a checkpoint using the consolidated approach.
    Now directly converts from torch checkpoints to HF format without dist2torch step.
    """
    print(f"Converting checkpoints using model config from: {log_path}")
    model_config = get_model_config_from_command_line(log_path)

    if not model_config:
        logging.debug(
            f"Skipping {log_path.name}, could not determine model configuration"
        )
        return

    # Get already converted iterations for this model
    converted_iterations = get_converted_iterations(save_checkpoints_dir)

    # Filter out iterations that have already been converted or are being converted
    remaining_iterations = []
    remaining_ckpt_iter_paths = []

    for iteration, path in zip(iterations, ckpt_iter_paths):
        marker_file = Path(save_checkpoints_dir) / "hf" / f"iter_{iteration}" / ".conversion_submitted"
        if iteration in converted_iterations:
            logging.info(f"Iteration {iteration} already converted, skipping")
        elif marker_file.exists():
            logging.info(f"Iteration {iteration} conversion already submitted, skipping")
        else:
            remaining_iterations.append(iteration)
            remaining_ckpt_iter_paths.append(path)

    # If no iterations need to be converted, return early
    if not remaining_iterations:
        logging.info(f"All iterations already converted or submitted, skipping")
        return

    print(f"Converting {len(remaining_iterations)} iterations")

    # Create necessary directories
    os.makedirs(convert_logs_dir, exist_ok=True)
    os.makedirs(save_checkpoints_dir, exist_ok=True)

    # Load the SBATCH template
    # get directory of this file
    cwd = Path(__file__).parent
    print(f"Using SBATCH template from {cwd}")

    sbatch_template_path =f"{cwd}/template.sbatch"

    try:
        with open(sbatch_template_path, "r") as f:
            sbatch_template = f.read()
            # escape ${} in f-strings with double curly braces
            # escape cat <<EOF > ../config.json\n{...}\nEOF
            cat_eof_data = re.search(
                r"cat <<EOF.*?EOF", sbatch_template, re.DOTALL
            ).group()
            sbatch_template = sbatch_template.replace(cat_eof_data, "<cat_eof_data>")
            sbatch_template = re.sub(
                r"\$\{(.+?)\}", r"\${{\1}}", sbatch_template
            ).replace("\$", "$")

        # Process each remaining iteration
        for iteration, path in zip(remaining_iterations, remaining_ckpt_iter_paths):
            logging.info(f"Converting iteration {iteration}")

            pre_run_cmd = f'export HF_HOME="{tokenizer_cache_dir}"'

            sbatch_script = sbatch_template.format(
                current_dir=cwd,
                account=account,
                partition=partition,
                container_image=container_image,
                torch_ckpt_path=checkpoint_dir,
                opensci_megatron_path=opensci_megatron_path,
                opensci_hf_path=opensci_hf_path,
                train_logs_path=str(log_path),
                save_checkpoints_dir=save_checkpoints_dir,
                convert_logs_dir=convert_logs_dir,
                pre_run_cmd=pre_run_cmd,
                venv_path=venv_path,
                iteration_to_convert=iteration,
                num_layers=model_config["NUM_LAYERS"],
                num_attn_heads=model_config["NUM_ATTN_HEADS"],
                ffn_hidden_size=model_config["FFN_HIDDEN_SIZE"],
                max_seq_length=model_config["MAX_POSITION_EMBEDDINGS"],
            )

            sbatch_script = sbatch_script.replace("<cat_eof_data>", cat_eof_data)

            # Create a temporary file for the sbatch script
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sbatch", delete=False
            ) as f:
                f.write(sbatch_script)
                sbatch_script_path = f.name

            subprocess.run(["sbatch", sbatch_script_path])
            logging.info(f"Submitted job for iteration {iteration}")

            # Create marker file to indicate job submission
            marker_dir = Path(save_checkpoints_dir) / "hf" / f"iter_{iteration}"
            marker_dir.mkdir(parents=True, exist_ok=True)
            marker_file = marker_dir / ".conversion_submitted"
            marker_file.touch()

            # Clean up the temporary file
            os.unlink(sbatch_script_path)

    except Exception as e:
        logging.error(f"Error converting checkpoint: {e}")
        logging.info(f"Skipping checkpoint: {log_path}")


def process_all_checkpoints_consolidated(
    checkpoint_dir: Union[str, Path],
    log_path: Union[str, Path],
    save_checkpoints_dir: str,
    opensci_megatron_path: str,
    opensci_hf_path: str,
    convert_logs_dir: str,
    account: str,
    partition: str,
    container_image: str,
    tokenizer_cache_dir: str,
    venv_path: Union[str, Path],
) -> None:
    """
    Process all checkpoints found in the logs using the consolidated approach.

    Args:
        checkpoint_dir: Directory containing checkpoints
        log_path: Path to the SLURM log file
        save_checkpoints_dir: Directory to save converted checkpoints
        opensci_megatron_path: Path to Megatron-LM-Open-Sci repository
        opensci_hf_path: Path to Open-Sci-hf repository
        convert_logs_dir: Directory to save conversion logs
        account: Account to use for conversion
        partition: Partition to use for conversion
        container_image: Container image to use for conversion
        tokenizer_cache_dir: Directory for tokenizer cache
    """
    print("Processing all checkpoints using consolidated workflow...")

    checkpoint_dir = Path(checkpoint_dir)
    log_path = Path(log_path)

    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise ValueError(
            f"Checkpoint directory {checkpoint_dir} does not exist or is not a directory"
        )

    print(f"\nFound {len(list(checkpoint_dir.iterdir()))} checkpoints")

    # Process each checkpoint
    checkpoint_iterations, ckpt_iter_paths = get_iterations_from_checkpoint(
        checkpoint_dir
    )

    # If no iterations found, skip this checkpoint
    if not len(checkpoint_iterations) > 0:
        logging.info(f"No iterations found in {checkpoint_dir}, skipping")

    convert_checkpoint_consolidated(
        checkpoint_dir=checkpoint_dir,
        venv_path=venv_path,
        log_path=log_path,
        iterations=checkpoint_iterations,
        ckpt_iter_paths=ckpt_iter_paths,
        save_checkpoints_dir=save_checkpoints_dir,
        opensci_megatron_path=opensci_megatron_path,
        opensci_hf_path=opensci_hf_path,
        convert_logs_dir=convert_logs_dir,
        account=account,
        partition=partition,
        container_image=container_image,
        tokenizer_cache_dir=tokenizer_cache_dir,
    )


def main():
    """Main function that handles command line arguments and runs the consolidated workflow."""
    parser = argparse.ArgumentParser(
        description="Consolidated checkpoint conversion workflow"
    )

    # Arguments from original checkpoint_conversion_workflow.py
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing torch checkpoints.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to the SLURM log file used for training the model.",
    )
    parser.add_argument(
        "--save_checkpoints_dir",
        type=str,
        required=True,
        help="Directory to save converted checkpoints",
    )
    parser.add_argument(
        "--opensci_megatron_path",
        type=str,
        required=True,
        help="Path to Megatron-LM-Open-Sci repository",
    )
    parser.add_argument(
        "--opensci_hf_path",
        type=str,
        required=True,
        help="Path to Open-Sci-hf repository",
    )
    parser.add_argument(
        "--convert_logs_dir",
        type=str,
        required=True,
        help="Directory to save conversion logs",
    )
    parser.add_argument(
        "--account",
        type=str,
        default="AIFAC_L01_028",
        help="Account to use for conversion",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="boost_usr_prod",
        help="Partition to use for conversion",
    )
    parser.add_argument(
        "--container_image",
        type=str,
        required=True,
        help="Container image to use for conversion",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default="/leonardo_work/AIFAC_L01_028/.cache",
        help="Directory for tokenizer cache.",
    )
    parser.add_argument(
        "--venv_path",
        type=str,
        required=True,
        help="Path to the virtual environment.",
    )

    args = parser.parse_args()

    # Run the consolidated workflow
    process_all_checkpoints_consolidated(
        venv_path=args.venv_path,
        checkpoint_dir=args.checkpoint_dir,
        log_path=args.log_path,
        save_checkpoints_dir=args.save_checkpoints_dir,
        opensci_megatron_path=args.opensci_megatron_path,
        opensci_hf_path=args.opensci_hf_path,
        convert_logs_dir=args.convert_logs_dir,
        account=args.account,
        partition=args.partition,
        container_image=args.container_image,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
    )


if __name__ == "__main__":
    main()
