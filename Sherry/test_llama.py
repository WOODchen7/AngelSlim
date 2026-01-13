# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Modified from ParetoQ，https://arxiv.org/abs/2502.02631

import math
import pdb

from transformers import AutoModelForCausalLM

import copy
import torch
import transformers
from utils import utils
from utils import datautils
from utils.datautils_e2e import make_data_module

from utils.process_args import process_args
from torch import distributed as dist
from transformers import default_data_collator, Trainer

log = utils.get_logger("clm")


def test():
    args, training_args = process_args()
    if args.report_to == "wandb":
        import wandb
        wandb.init(project="General-QAT", name=args.run_name, config=args)

    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    assert args.model_family in ["llama", "qwen"], "Model family must be in ['llama', 'qwen']"

    if args.model_family == "llama":
        from models.llama.configuration_llama import LlamaConfig
        from models.llama.modeling_llama_quant import (
            LlamaForCausalLM as CausalLMQuant,
        )
        config = LlamaConfig.from_pretrained(args.model_path)
    elif args.model_family == "qwen":
        from models.qwen3.configuration_qwen3 import Qwen3Config
        from models.qwen3.modeling_qwen3_quant import (
            Qwen3ForCausalLM as CausalLMQuant,
        )
        config = Qwen3Config.from_pretrained(args.model_path)

    config.w_bits = args.w_bits
    config.quant_method = args.quant_method
    config.granularity = args.granularity
    config.group_size = args.group_size
    config.enable_zero_point = args.enable_zero_point
    config.range_of_lambada = args.range_of_lambada
    config.eps = args.eps
    config.N = args.N
    config.M = args.M

    model = CausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )

    if not args.contain_weight_clip_val:
        for name, param in model.named_parameters():
            if "weight_clip_val" in name:
                weight_name = name.replace("weight_clip_val", "weight")
                weight_param = dict(model.named_parameters()).get(weight_name, None)

                if args.w_bits == 1:
                    scale = torch.mean(weight_param.abs(), dim=-1, keepdim=True).detach()
                elif args.w_bits == 0 or args.w_bits == 2:
                    scale, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                elif args.w_bits == 3 or args.w_bits == 4:
                    xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    maxq = 2 ** (args.w_bits - 1) - 1
                    scale = xmax / maxq
                else:
                    raise NotImplementedError
                param.data.copy_(scale)

    model.cuda()
    log.info("Complete model loading...")

    if args.eval_tasks != "" or args.do_mmlu_eval:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        from lm_eval.tasks import TaskManager

    if args.eval_tasks != "":
        task_list = args.eval_tasks.split(',')
        lm_eval_model = HFLM(pretrained=model, batch_size=32)
        task_manager = TaskManager()
        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model=lm_eval_model,
            tasks=task_list,
            num_fewshot=0,
            task_manager=task_manager,
        )
        log.info(make_table(results))
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
            if args.report_to == "wandb":
                wandb.log({f'eval/{task}_acc': results['results'][task]['acc,none']})
        log.info(f'Average Acc: {total_acc / len(task_list) * 100:.2f}%')

    if args.do_mmlu_eval:
        lm_eval_model = HFLM(pretrained=model, batch_size=16)
        task_manager = TaskManager()
        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model=lm_eval_model,
            tasks=['mmlu'],
            num_fewshot=5,
            task_manager=task_manager,
            cache_requests=True,
        )
        log.info(make_table(results))
        total_acc = 0
        for task in results['results']:
            total_acc += results['results'][task]['acc,none']
            if args.report_to == "wandb":
                wandb.log({f'eval/{task}_acc': results['results'][task]['acc,none']})
        log.info(f"Average MMLU Acc: {total_acc / len(results['results']) * 100:.2f}%")


if __name__ == "__main__":
    test()
