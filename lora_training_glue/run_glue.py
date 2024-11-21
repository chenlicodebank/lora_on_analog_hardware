#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# LoRA related imports
from peft import LoraConfig
from peft import get_peft_model
from peft import PeftModel
from torch import save as torch_save, load as torch_load

# other related imports
from related_functions import list_analog_linear_layers, list_linear_layers, replace_layer, convert_selected_layers_to_analog
import torch

# - AIHWKIT related imports
import aihwkit
from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import AnalogSequential
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.tiles.inference_torch import TorchInferenceTile
from aihwkit.simulator.tiles.inference import InferenceTile
from aihwkit.simulator.presets.utils import IOParameters
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.noise.neurosoc_lamina import NeuroSoCLaminaModel
from aihwkit.inference.noise.neurosoc_standard import NeuroSoCStandardModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.simulator.configs import InferenceRPUConfig, TorchInferenceRPUConfig
from aihwkit.simulator.presets import PresetIOParameters
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    BoundManagementType,
    WeightClipType,
    NoiseManagementType,
    WeightRemapType,
)
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    WeightModifierType,
    WeightClipType,
    WeightNoiseType,
    BoundManagementType,
    NoiseManagementType,
    WeightClipParameter,
    WeightModifierParameter,
    MappingParameter,
)

# non huggingface parser import
import argparse

import wandb
wandb.login()
# Get the current folder name
current_folder_name = os.path.basename(os.getcwd())
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="lora_on_analog_hardware",

    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"{current_folder_name}",
    # track hyperparameters and run metadata
)

def gen_rpu_config(output_noise_level=0.04, pcm_model="NeuroSoCStandard_Gmax20"):
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    # rpu_config.modifier.per_batch_sample = False
    rpu_config.modifier.std_dev = 0.067
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
    # rpu_config.modifier.type = WeightModifierType.MULT_NORMAL
    rpu_config.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC
    # rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC
    rpu_config.forward = IOParameters()
    rpu_config.forward.out_noise = output_noise_level
    rpu_config.forward.is_perfect = False
    # rpu_config.modifier.rel_to_actual_wmax = True
    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu_config.clip.sigma = 3
    # rpu_config.clip.type = WeightClipType.FIXED_VALUE
    # rpu_config.clip.fixed_value = 1.0
    # rpu_config.forward.inp_res = -1
    # rpu_config.forward.out_res = -1
    rpu_config.forward.inp_res = 1 / (2**8 - 2)  # 8-bit resolution.
    rpu_config.forward.out_res = 1 / (2**8 - 2)  # 8-bit resolution.
    # rpu_config.forward.out_bound = 1000000
    # rpu_config.forward.inp_bound = 1
    # rpu_config.forward.bound_management = BoundManagementType.NONE
    # rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX
    # rpu_config.pre_post.input_range.enable = False

    if pcm_model == "NeuroSoCStandard_Gmax20":
        rpu_config.noise_model = NeuroSoCStandardModel(g_max=20)
    elif pcm_model == "NeuroSoCStandard_Gmax55":
        rpu_config.noise_model = NeuroSoCStandardModel(g_max=55)
    elif pcm_model == "NeuroSoCLamina_Gmax20":
        rpu_config.noise_model = NeuroSoCLaminaModel(g_max=20)
    elif pcm_model == "NeuroSoCLamina_Gmax55":
        rpu_config.noise_model = NeuroSoCLaminaModel(g_max=55)
    elif pcm_model == "PCM_Gmax25":
        rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    else:
        raise ValueError(f"Unknown PCM model: {pcm_model}")

    return rpu_config

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.46.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/0_4_v1/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():
    # Parse additional PCM model and noise level arguments
    general_parser = argparse.ArgumentParser(description="General parser for PCM model and noise level")
    general_parser.add_argument('--pcm_model', type=str, default="NeuroSoCStandard_Gmax20", help="PCM model to use")
    general_parser.add_argument('--output_noise_level', type=float, default=0.04, help="Noise level to use in RPU config")
    general_parser.add_argument('--analog_optimizer', type=str, default=None,
                                help="Analog optimizer to use (options: 'AnalogSGD', 'AnalogAdam'). If not provided, no analog optimizer is used.")
    general_parser.add_argument('--analog_lr', type=float, default=0.0005,
                                help="Learning rate for the analog optimizer")
    general_parser.add_argument('--num_evaluation_drift_values', type=int, default=7,
                                help="Number of drift values to evaluate")
    general_parser.add_argument('--num_evaluation_repetition', type=int, default=1,
                                help="Number of times to repeat the evaluation")

    general_args, remaining_argv = general_parser.parse_known_args()

    # Remove the parsed arguments from sys.argv
    sys.argv = [sys.argv[0]] + remaining_argv

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.label_names = ['labels']
    print("training_args")
    print(training_args)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "nyu-mll/glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Print the parsed customized arguments
    print("Parsed Arguments:")
    print(f"PCM Model: {general_args.pcm_model}")
    print(f"Output Noise Level: {general_args.output_noise_level}")
    print(f"Analog Optimizer: {general_args.analog_optimizer}")
    print(f"Analog Learning Rate: {general_args.analog_lr}")
    print(f"Number of Evaluation Drift Values: {general_args.num_evaluation_drift_values}")
    print(f"Number of Evaluation Repetition: {general_args.num_evaluation_repetition}")


    print("Original digital model:")
    print(model)



    # peft_config = LoraConfig(r=8, lora_alpha=32,
    #                          lora_dropout=0.pcm_adcnoise0,
    #                          target_modules=["dense"],)
    peft_config = LoraConfig(r=8, lora_alpha=32,
                             lora_dropout=0.1,
                             target_modules=["dense","query","key","value","qa_outputs","embedding_transformation"],)


    if training_args.do_train:
        model = get_peft_model(model, peft_config)
        print("Digital model with LORA:")
        model.print_trainable_parameters()
    elif training_args.do_eval:
        # model = PeftModel.from_pretrained(model, "./squad_models_train")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        raise ValueError("command not found!")
    print("Digital model with LORA")
    print(model)
    digital_model = model

    model = convert_to_analog(model, gen_rpu_config(output_noise_level=general_args.output_noise_level,
                                                    pcm_model=general_args.pcm_model),
                              tile_module_class=TorchInferenceTile)
    print("RPU Configuration:")
    print(gen_rpu_config(output_noise_level=general_args.output_noise_level, pcm_model=general_args.pcm_model))

    print("Analog model with LoRA (Before layer correction)")
    print(model)


    analog_linear_layer_names = list_analog_linear_layers(model)

    # # print analog linear layer names for debugging
    # print("analog_linear_layer_names")
    # print(analog_linear_layer_names)


    # substrings_to_remove = ["query", "key", "value", "dense", 'mobilebert.embeddings.embedding_transformation', 'qa_outputs']
    substrings_to_remove = ["base_layer"]

    filtered_analog_linear_layer_names = [name for name in analog_linear_layer_names
                                          if not any(substring in name for substring in substrings_to_remove)]


    for layer_name in filtered_analog_linear_layer_names:
        replace_layer(model, digital_model, layer_name)


    if training_args.do_train:
        model.print_trainable_parameters()
    model.to("cuda")
    print("Analog model with LORA (After layer correction)")
    print(model)


    for module in model.modules():
        if isinstance(module, AnalogLinear):  # Check if the module is an instance of AnalogLinear
            for param in module.parameters():
                param.requires_grad = False
    if training_args.do_train:
        model.print_trainable_parameters()
    # if training_args.do_eval:
    #     model.load_state_dict(torch_load("saved_chkpt.pt"))
    #     model.eval()
    #     model.drift_analog_weights(1000000)
    #     # model = model.module
    #     print("analog model for inference")
    #     print(model)

    if general_args.analog_optimizer == "AnalogSGD":
        optimizer = AnalogSGD(model.parameters(), lr=general_args.analog_lr, momentum=0.9)
    elif general_args.analog_optimizer == "AnalogAdam":
        optimizer = AnalogAdam(model.parameters(), lr=general_args.analog_lr)
    else:
        optimizer = None




    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif is_regression:
        metric = evaluate.load("mse", cache_dir=model_args.cache_dir)
    else:
        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        optimizers=(optimizer, None),  # Pass the analog optimizer
        # load_best_model_at_end=True,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.save_model()  # Saves the tokenizer too for easy upload
        # torch_save(model.state_dict(), "./saved_chkpt.pt")

        output_model_file = os.path.join(training_args.output_dir, "saved_chkpt.pt")
        torch.save(model.state_dict(), output_model_file)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #
    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     tasks = [data_args.task_name]
    #     eval_datasets = [eval_dataset]
    #     if data_args.task_name == "mnli":
    #         tasks.append("mnli-mm")
    #         valid_mm_dataset = raw_datasets["validation_mismatched"]
    #         if data_args.max_eval_samples is not None:
    #             max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
    #             valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
    #         eval_datasets.append(valid_mm_dataset)
    #         combined = {}
    #
    #     for eval_dataset, task in zip(eval_datasets, tasks):
    #         metrics = trainer.evaluate(eval_dataset=eval_dataset)
    #         print("metrics")
    #         print(metrics)
    #         max_eval_samples = (
    #             data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #         )
    #         metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    #
    #         if task == "mnli-mm":
    #             metrics = {k + "_mm": v for k, v in metrics.items()}
    #         if task is not None and "mnli" in task:
    #             combined.update(metrics)
    #
    #         trainer.log_metrics("eval", metrics)
    #         trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)
    #
    # # Evaluation
    # if training_args.do_eval:
    #     # Define all possible drift values (0 second, 1 hour, 1 day, 1 week, 1 month, 1 year, 10 years)
    #     all_drift_values = [0, 3600, 86400, 604800, 2592000, 31536000, 315360000]
    #
    #     # Select the first `num_evaluation_drift_values` drift values
    #     drift_values = all_drift_values[:general_args.num_evaluation_drift_values]
    #
    #     logger.info("*** Evaluate ***")
    #
    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     tasks = [data_args.task_name]
    #     eval_datasets = [eval_dataset]
    #     if data_args.task_name == "mnli":
    #         tasks.append("mnli-mm")
    #         valid_mm_dataset = raw_datasets["validation_mismatched"]
    #         if data_args.max_eval_samples is not None:
    #             max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
    #             valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
    #         eval_datasets.append(valid_mm_dataset)
    #         combined = {}
    #
    #     eval_metrics = {}
    #
    #     for drift in drift_values:
    #         all_metrics = []
    #
    #         for i in range(general_args.num_evaluation_repetition):
    #             # Load model checkpoint before each evaluation
    #             # model.load_state_dict(torch.load('saved_chkpt.pt'), load_rpu_config=False)
    #
    #             output_model_file = os.path.join(training_args.output_dir, "saved_chkpt.pt")
    #             model.load_state_dict(torch.load(output_model_file), load_rpu_config=False)
    #
    #             model.eval()
    #
    #             print(
    #                 f"Testing with drift_analog_weights = {drift} (Repetition {i + 1}/{general_args.num_evaluation_repetition})")
    #
    #             model.drift_analog_weights(drift)
    #
    #             for eval_dataset, task in zip(eval_datasets, tasks):
    #                 metrics = trainer.evaluate(eval_dataset=eval_dataset)
    #                 print("metrics")
    #                 print(metrics)
    #                 max_eval_samples = (
    #                     data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #                 )
    #                 metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    #
    #                 if task == "mnli-mm":
    #                     metrics = {k + "_mm": v for k, v in metrics.items()}
    #                 if task is not None and "mnli" in task:
    #                     combined.update(metrics)
    #
    #             all_metrics.append(metrics)
    #
    #         # Calculate mean and standard deviation for the metrics
    #         mean_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    #         std_metrics = {key: np.std([m[key] for m in all_metrics]) for key in all_metrics[0]}
    #
    #         eval_metrics[drift] = {'mean': mean_metrics, 'std': std_metrics}
    #
    #         # Save the metrics
    #         trainer.log_metrics(f"eval_mean_driftsecond={drift}s", mean_metrics)
    #         trainer.save_metrics(f"eval_mean_driftsecond={drift}s", mean_metrics)
    #         trainer.log_metrics(f"eval_std_driftsecond={drift}s", std_metrics)
    #         trainer.save_metrics(f"eval_std_driftsecond={drift}s", std_metrics)
    #         print(
    #             f"Drift_analog_weights = {drift} second")
    #
    #         # Prepare the key with the formatted drift value
    #         key_name = f"eval_mean_vs_drift_seconds"
    #
    #         # Log the metric to wandb
    #         wandb.log({key_name: mean_metrics})

    # Evaluation
    if training_args.do_eval:
        # Define all possible drift values (0 second, 1 hour, 1 day, 1 week, 1 month, 1 year, 10 years)
        all_drift_values = [0, 3600, 86400, 604800, 2592000, 31536000, 315360000]

        # Select the first num_evaluation_drift_values drift values
        drift_values = all_drift_values[:general_args.num_evaluation_drift_values]

        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        eval_metrics = {}

        for drift in drift_values:
            all_metrics = []

            for i in range(general_args.num_evaluation_repetition):
                # Load model checkpoint before each evaluation
                output_model_file = os.path.join(training_args.output_dir, "saved_chkpt.pt")
                model.load_state_dict(torch.load(output_model_file), load_rpu_config=False)

                model.eval()

                print(
                    f"Testing with drift_analog_weights = {drift} (Repetition {i + 1}/{general_args.num_evaluation_repetition})"
                )

                model.drift_analog_weights(drift)

                # Collect metrics for all tasks
                metrics_per_repetition = {}

                for eval_dataset, task in zip(eval_datasets, tasks):
                    metrics = trainer.evaluate(eval_dataset=eval_dataset)
                    print("metrics")
                    print(metrics)
                    max_eval_samples = (
                        data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
                    )
                    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

                    if task == "mnli-mm":
                        metrics = {k + "_mm": v for k, v in metrics.items()}
                    if task is not None and "mnli" in task:
                        combined.update(metrics)

                    # Update metrics_per_repetition with metrics from this task
                    metrics_per_repetition.update(metrics)

                # Append the metrics for all tasks from this repetition
                all_metrics.append(metrics_per_repetition)

            # Calculate mean and standard deviation for the metrics across all repetitions
            keys = all_metrics[0].keys()
            mean_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in keys}
            std_metrics = {key: np.std([m[key] for m in all_metrics]) for key in keys}

            eval_metrics[drift] = {'mean': mean_metrics, 'std': std_metrics}

            # Save the metrics
            trainer.log_metrics(f"eval_mean_driftsecond={drift}s", mean_metrics)
            trainer.save_metrics(f"eval_mean_driftsecond={drift}s", mean_metrics)
            trainer.log_metrics(f"eval_std_driftsecond={drift}s", std_metrics)
            trainer.save_metrics(f"eval_std_driftsecond={drift}s", std_metrics)
            print(f"Drift_analog_weights = {drift} second")

            # Prepare the key with the formatted drift value
            key_name = f"eval_mean_vs_drift_seconds"

            # Log the metric to wandb
            wandb.log({key_name: mean_metrics})

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "0_4_v1"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
