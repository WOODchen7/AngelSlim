# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Modified from ParetoQ，https://arxiv.org/abs/2502.02631

import os
from dataclasses import dataclass, field
from typing import Optional
import argparse
import transformers


@dataclass
class ModelArguments:
    local_dir: str = field(
        default="./output", metadata={"help": "Local Path of storing inputs and outputs "}
    )
    model_path: Optional[str] = field(
        default=None, metadata={"help": "Input model relative manifold path"}
    )
    output_model_filename: Optional[str] = field(
        default=None, metadata={"help": "Output model relative manifold path"}
    )
    model_family: Optional[str] = field(
        default="llama-3.2-1B",
        metadata={"help": "for the saving of dataset cache for faster experiments"}
    )
    output_model_local_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )
    w_bits: Optional[int] = field(
        default=32,
        metadata={
            "help": "#bits to use for quantization; use 16 for evaluating base model. choices=[4, 8, 32]"
        },
    )
    group_size: Optional[int] = field(
        default=128,
        metadata={
            "help": "Group size to use for quantization; default is 128. choices=[-1, 32, 64, 128]"
        },
    )
    enable_zero_point: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable zero point quantization."}
    )

    N: Optional[int] = field(
        default=3,
        metadata={"help": "N for N:M sparse"}
    )

    M: Optional[int] = field(
        default=4,
        metadata={"help": "M for N:M sparse"}
    )

    quant_method: Optional[str] = field(
        default="lsq", metadata={"help": "Quantization method to use. choices=[absmean, twn, quaternary_static, quaternary_lsq, lsq]"}
    )
    granularity: Optional[str] = field(
        default="per_tensor",
        metadata={"help": "Granularity to use for quantization. choices=[per_tensor, per_channel]"}
    )
    contain_weight_clip_val: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set contain_weight_clip_val=True when load a trained quantized model."
        },
    )
    my_lr_scheduler_kwargs: Optional[str] = field(
        default="{}",
        metadata={"help": "my_lr_scheduler_kwargs"}
    )


# @dataclass
# class DataArguments:
#     max_train_samples: Optional[int] = field(
#         default=-1,
#         metadata={
#             "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
#             "value if set."
#         },
#     )
#     max_eval_samples: Optional[int] = field(
#         default=-1,
#         metadata={
#             "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
#             "value if set."
#         },
#     )


@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    dataset_root: str = field(
        default='/data/datasets/',
        metadata={"help": "Root directory of the dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    eval_tasks: str = field(
        default='',
        metadata={"help": "evaluation tasks for lm eval, example:piqa,arc_easy,arc_challenge,hellaswag,winogrande"}
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    # dataset_path: str = field(
    #     default=None,
    #     metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    # )
    eval_tasks: str = field(
        default='',
        metadata={"help": "evaluation tasks for lm eval, example:piqa,arc_easy,arc_challenge,hellaswag,winogrande"}
    )
    conv_temp: str = field(
        default='llama-2',
        metadata={"help": "Conversation template, only useful with deita datasets"}
    )
    mask_use: bool = field(
        default=True, metadata={"help": "mask the loss to role in dialogue datas"}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|redpajama]"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=96,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    predict_with_generate: Optional[bool] = field(
        default=False
    )

    train_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Train data local path"}
    )
    eval_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Eval data local path"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="./output/")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    qat: Optional[bool] = field(default=False)
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    pt_context_len: int = field(
        default=4096,
        metadata={"help": "language modeling length."}
    )
    remove_unused_columns: Optional[bool] = field(
        default=False,
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )

    begin_train_ratio: Optional[float] = field(
        default=0.0,
        metadata={"help": "The ratio of the training data to use for training."}
    )

    range_of_lambada: Optional[float] = field(
        default=0.01,
        metadata={"help": "The range of lambada in tequlila_v2 and tequila_v3."}
    )

    eps: Optional[float] = field(
        default=1e-3,
        metadata={"help": "The range of lambada in ultraquantv3."}
    )

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


def process_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    model_args, data_args, training_args, generation_args = parser.parse_args_into_dataclasses()

    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))

    os.makedirs(model_args.local_dir, exist_ok=True)

    model_name = model_args.model_family

    # determine the detailed saving dir
    # postfix = f"{training_args.wbits}b_{training_args.quant_method}_{training_args.granularity}_{training_args.group_size}g_{training_args.enable_zero_point}(zero)"
    postfix = f"{model_args.w_bits}b_{model_args.quant_method}_{model_args.granularity}_{model_args.group_size}g"

    assert model_args.output_model_local_path is None

    model_args.output_model_local_path = os.path.join(
        model_args.local_dir, data_args.dataset, f"{str(model_args.output_model_filename)}-{postfix}"
    )
    training_args.output_dir = os.path.join(
        model_args.local_dir, data_args.dataset, f"{str(model_args.output_model_filename)}-{postfix}"
    )
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    training_args.run_name = f"{model_name}-{data_args.dataset}-{postfix}"

    return args, training_args
