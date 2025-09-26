# Copyright 2025 Tencent Inc. All Rights Reserved.
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

import copy
import os
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import transformers
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.nn.utils.rnn import pad_sequence
from transformers import default_data_collator

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [
            f"{self.tokenizer.bos_token}{example['input']}" for example in instances
        ]
        targets = [
            f"{example['output']}{self.tokenizer.eos_token}" for example in instances
        ]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
            padding="max_length",
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
            padding="max_length",
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt["input_ids"], tokenized_targets["input_ids"]
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + copy.deepcopy(tokenized_target)
                        )
                    )
                else:
                    labels.append(
                        torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
                    )
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict


ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {"input": prompt_format.format(**example)}


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning or continue pre-train.
    """

    dataset = None
    def load_data(dataset_name):
        if dataset_name == "alpaca":
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == "oasst1":
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == "deita-6k":
            dataset = load_dataset("hkust-nlp/deita-6k-v0", split="train")
            dataset = [row for row in dataset]
            return dataset
        elif dataset_name == "deita-10k":
            dataset = load_dataset("hkust-nlp/deita-10k-v0", split="train")
            dataset = [row for row in dataset]
            return dataset
        elif dataset_name == "c4":
            try:
                # load from local file, a fast manner
                dataset = load_dataset(
                    "arrow",
                    data_files={
                        "train": "",
                        "validation": "",
                    },
                )
            except:
                dataset = load_dataset(
                    "allenai/c4",
                    "allenai--c4",
                    data_files={
                        "train": "en/c4-train.00000-of-01024.json.gz",
                        "validation": "en/c4-validation.00000-of-00008.json.gz",
                    },
                )
            return dataset
        elif dataset_name == "redpajama-sample":
            try:
                loacal_dataset = (
                    f"{args.dataset_root}/datasets/RedPajama-Data-1T-Sample/"
                )
                dataset = load_from_disk(loacal_dataset)
            except:
                dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")

        elif dataset_name == "wiki-10b":
            local_dataset = f"{args.dataset_root}/datasets/wiki-10B"
            dataset = load_from_disk(local_dataset)
        elif dataset_name == "redpajama-1b":
            raise NotImplementedError
        elif dataset_name in ["redpajama-2b"]:
            local_dataset = f"{args.dataset_root}/datasets/RedPajama-Data-Arxiv-2B"
            dataset = load_from_disk(local_dataset)
        elif dataset_name in ["redpajama-10b-arxiv"]:
            local_dataset = f"{args.dataset_root}/datasets/RedPajama-Data-Arxiv-10B/"
            dataset = load_from_disk(local_dataset)
        elif dataset_name in ["redpajama-10b"]:
            local_dataset = f"{args.dataset_root}/datasets/RedPajama-Data-C4-10B/"
            dataset = load_from_disk(local_dataset)
        elif dataset_name in ["ultrafineweb-10b"]:
            local_dataset = f"{args.dataset_root}/datasets/Ultra-FineWeb-10B/"
            dataset = load_from_disk(local_dataset)
        elif dataset_name in ["ultrafineweb-zh-10b"]:
            local_dataset = f"{args.dataset_root}/datasets/Ultra-FineWeb-High-zh/"
            dataset = load_from_disk(local_dataset)
        elif dataset_name in ["ultrafineweb-24b"]:
            local_dataset = f"{args.dataset_root}/datasets/Ultra-FineWeb-High-Conf-24B/"
            dataset = load_from_disk(local_dataset)
        elif dataset_name in ["ultrafineweb-highconf-12b"]:
            local_dataset = f"{args.dataset_root}/datasets/Ultra-FineWeb-High-Conf-12B/"
            dataset = load_from_disk(local_dataset)
        elif dataset_name in ["slimpajama-14b"]:
            local_dataset = f"{args.dataset_root}/datasets/SlimPajama-Subset-14B/"
            dataset = load_from_disk(local_dataset)
        elif dataset_name in ["redpajama-28b"]:
            loacal_dataset = f"{args.dataset_root}/datasets/RedPajama-Data-1T/"
            os.environ["RED_PAJAMA_DATA_DIR"] = loacal_dataset
            dataset = load_dataset("togethercomputer/RedPajama-Data-1T", "arxiv")
        elif dataset_name in ["redpajama-100b"]:
            loacal_dataset = f"{args.dataset_root}/datasets/RedPajama-Data-1T/"
            os.environ["RED_PAJAMA_DATA_DIR"] = loacal_dataset
            dataset = load_dataset("togethercomputer/RedPajama-Data-1T", "c4")
        elif dataset_name in ["redpajama-1t"]:
            try:
                loacal_dataset = f"{args.dataset_root}/datasets/RedPajama-Data-1T/"
                os.environ["RED_PAJAMA_DATA_DIR"] = loacal_dataset
                dataset = load_dataset("togethercomputer/RedPajama-Data-1T", "arxiv")
            except:
                dataset = load_dataset("togethercomputer/RedPajama-Data-1T", "arxiv")
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

        if not isinstance(dataset, DatasetDict):
            dataset = DatasetDict({"train": dataset})

        if "validation" not in dataset.keys():
            validation_split = args.eval_dataset_size
            dataset["validation"] = dataset["train"].select(range(validation_split))

        return dataset

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == "alpaca"
            or dataset_format == "alpaca-clean"
            or (dataset_format is None and args.dataset in ["alpaca", "alpaca-clean"])
        ):
            dataset = dataset.map(
                extract_alpaca_dataset, remove_columns=["instruction"]
            )
        elif dataset_format == "oasst1" or (
            dataset_format is None and args.dataset == "oasst1"
        ):
            dataset = dataset.map(
                lambda x: {
                    "input": "",
                    "output": x["text"],
                }
            )
        elif dataset_format == "pt" or (
            dataset_format is None
            and args.dataset
            in [
                "c4",
                "redpajama-sample",
                "redpajama-1b",
                "redpajama-2b",
                "redpajama-10b",
                "ultrafineweb-10b",
                "ultrafineweb-zh-10b",
                "ultrafineweb-24b",
                "ultrafineweb-highconf-12b",
                "slimpajama-14b",
                "redpajama-10b-arxiv",
                "redpajama-28b",
                "redpajama-100b",
                "redpajama-1t",
            ]
        ):
            block_size = args.pt_context_len
            column_names = list(dataset["train"].features)
            text_column_name = "text" if "text" in column_names else column_names[0]

            def tokenize_function(examples):
                output = tokenizer(
                    examples[text_column_name],
                    padding="max_length",
                    truncation=True,
                    max_length=block_size,
                )
                return output

            tokenized_datasets = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
                num_proc=os.cpu_count(),
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: list(chain(*examples[k])) for k in examples.keys()
                }
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                if total_length >= block_size:
                    total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                result = {
                    k: [
                        t[i : i + block_size]
                        for i in range(0, total_length, block_size)
                    ]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            dataset = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=os.cpu_count(),
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        # Remove unused columns for instruction-tuning
        if not dataset_format == "pt":
            dataset = dataset.remove_columns(
                [
                    col
                    for col in dataset.column_names["train"]
                    if col not in ["input", "output"]
                ]
            )
        return dataset

    # Load dataset.
    print(f"loading {args.dataset}")
    if args.dataset in [
        "c4",
        "wiki-10b",
        "redpajama-sample",
        "redpajama-1b",
        "redpajama-2b",
        "redpajama-10b",
        "ultrafineweb-10b",
        "ultrafineweb-zh-10b",
        "ultrafineweb-24b",
        "ultrafineweb-highconf-12b",
        "slimpajama-14b",
        "redpajama-10b-arxiv",
        "redpajama-28b",
        "redpajama-100b",
        "redpajama-1t",
    ]:
        cache_dir = f"{args.dataset_root}/dataset_cache"
        cache_dataloader = f"{cache_dir}/{args.dataset}_{args.pt_context_len}_cache/"
        print(f"cache_dataloader: {cache_dataloader}")
        if os.path.exists(cache_dataloader):
            dataset = load_from_disk(cache_dataloader)
            print(f"load dataset cache from {cache_dataloader}")
            if args.begin_train_ratio != 0.0:
                print(f"begining ratio {args.begin_train_ratio}")
                dataset["train"] = dataset["train"].select(
                    range(
                        int(args.begin_train_ratio * len(dataset["train"])),
                        len(dataset["train"]),
                    )
                )
        # elif args.dataset in ['redpajama-10b', "redpajama-10b-arxiv"]:
        #     cache_dataloader = f'{cache_dir}/{args.dataset}_2048_cache/'
        #     dataset = load_from_disk(cache_dataloader)
        #     dataset = dataset.map(lambda x: {
        #         'input_ids': x['input_ids'][:args.pt_context_len],
        #         'attention_mask': x['attention_mask'][:args.pt_context_len],
        #         'labels': x['labels'][:args.pt_context_len],
        #     }, num_proc=os.cpu_count(), batched=True)
        else:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            dataset = load_data(args.dataset)
            dataset = format_dataset(dataset, args.dataset_format)
            dataset.save_to_disk(cache_dataloader, num_proc=os.cpu_count())
        # dataset = load_data(args.dataset)
        # dataset = format_dataset(dataset, args.dataset_format)
    elif args.dataset in ["deita-6k", "deita-10k"]:
        # Split train/eval for deita datasets
        raw_data = load_data(args.dataset)
        np.random.seed(0)
        train_raw_data = raw_data
        perm = np.random.permutation(len(raw_data))
        split = int(len(perm) * 0.98)
        train_indices = perm[:split]
        eval_indices = perm[split:]
        train_raw_data = [raw_data[i] for i in train_indices]
        eval_raw_data = [raw_data[i] for i in eval_indices]
        print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")
        from deita_dataset.train import LazySupervisedDataset, SupervisedDataset

        dataset_cls = LazySupervisedDataset
        train_dataset = dataset_cls(
            train_raw_data,
            tokenizer=tokenizer,
            conv_template=args.conv_temp,
            mask_user=args.mask_use,
        )
        eval_dataset = dataset_cls(
            eval_raw_data,
            tokenizer=tokenizer,
            conv_template=args.conv_temp,
            mask_user=args.mask_use,
        )
    elif args.dataset == "mix_deita_redpajama":
        cache_dir = f"{args.dataset_root}/dataset_cache"
        cache_dataloader = f"{cache_dir}/{args.dataset}_{args.pt_context_len}_cache/"
        if os.path.exists(cache_dataloader):
            dataset = load_from_disk(cache_dataloader)
            print(f"load dataset from {cache_dataloader}")
        else:
            deita_dataset = load_data("deita-10k")
            np.random.seed(0)
            from datasets import Dataset, concatenate_datasets
            from utils.deita_train import SupervisedDataset

            print("tokenizr deita, need a long time.")
            deita_dataset = SupervisedDataset(
                deita_dataset,
                tokenizer=tokenizer,
                conv_template=args.conv_temp,
                mask_user=args.mask_use,
            )
            deita_dataset = Dataset.from_dict(
                {
                    "input_ids": deita_dataset.input_ids,
                    "labels": deita_dataset.labels,
                    "attention_mask": deita_dataset.attention_mask,
                }
            )
            dataset = load_data("redpajama-1b")
            redpajama_dataset = format_dataset(dataset, "pt")

            train_dataset = concatenate_datasets(
                [
                    deita_dataset,
                    redpajama_dataset["train"].select(range(len(deita_dataset))),
                ]
            )
            dataset = {
                "train": train_dataset,
                "validation": redpajama_dataset["validation"],
            }
            Path(cache_dataloader).mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(cache_dataloader)
    else:
        dataset = load_data(args.dataset)
        dataset = format_dataset(dataset, args.dataset_format)
    print(f"loading {args.dataset} successfully")

    # Split train/eval, reduce size for other datasets
    if not args.dataset in ["deita-6k", "deita-10k"]:
        if args.do_eval or args.do_predict:
            if "eval" in dataset:
                eval_dataset = dataset["eval"]
            elif "validation" in dataset:
                eval_dataset = dataset["validation"]
            else:
                print(
                    "Splitting train dataset in train and validation according to `eval_dataset_size`"
                )
                dataset = dataset["train"].train_test_split(
                    test_size=args.eval_dataset_size, shuffle=True, seed=42
                )
                eval_dataset = dataset["test"]
            if (
                args.max_eval_samples is not None
                and len(eval_dataset) > args.max_eval_samples
            ):
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))
            if args.group_by_length:
                eval_dataset = eval_dataset.map(
                    lambda x: {"length": len(x["input"]) + len(x["output"])}
                )
        if args.do_train:
            train_dataset = dataset["train"]
            train_dataset = train_dataset.shuffle(seed=0)
            if (
                args.max_train_samples is not None
                and len(train_dataset) > args.max_train_samples
            ):
                train_dataset = train_dataset.select(range(args.max_train_samples))
            if args.group_by_length:
                train_dataset = train_dataset.map(
                    lambda x: {"length": len(x["input"]) + len(x["output"])}
                )
    if args.dataset in [
        "c4",
        "wiki-10b",
        "redpajama-1b",
        "redpajama-2b",
        "redpajama-10b",
        "redpajama-10b-arxiv",
        "ultrafineweb-10b",
        "ultrafineweb-zh-10b",
        "ultrafineweb-24b",
        "ultrafineweb-highconf-12b",
        "slimpajama-14b",
        "redpajama-28b",
        "redpajama-100b",
        "redpajama-1t",
        "deita-6k",
        "deita-10k",
        "mix_deita_redpajama",
    ]:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            train_on_source=args.train_on_source,
            predict_with_generate=args.predict_with_generate,
        )

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator,
    )
