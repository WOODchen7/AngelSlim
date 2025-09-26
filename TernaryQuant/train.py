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

import json

import torch
import transformers
from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant
from torch import distributed as dist
from transformers import Trainer
from utils import utils
from utils.datautils_e2e import make_data_module
from utils.process_args import process_args

log = utils.get_logger("clm")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def train():
    dist.init_process_group(backend="nccl")
    args, training_args = process_args()
    print(training_args.lr_scheduler_kwargs)
    print(
        json.loads(args.my_lr_scheduler_kwargs),
        type(json.loads(args.my_lr_scheduler_kwargs)),
    )
    training_args.lr_scheduler_kwargs = json.loads(args.my_lr_scheduler_kwargs)
    if args.report_to == "wandb":
        import wandb

        wandb.init(project="General-QAT", name=args.run_name, config=args)

    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    config = LlamaConfig.from_pretrained(args.model_path)
    config.w_bits = args.w_bits
    config.quant_method = args.quant_method
    config.granularity = args.granularity
    config.group_size = args.group_size
    config.enable_zero_point = args.enable_zero_point
    config.range_of_lambada = args.range_of_lambada
    config.eps = args.eps

    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cuda",  # 必须是cuda !!! 不然会出现设备转移的nan问题
    )

    # for name, param in model.named_parameters():
    #     param.requires_grad = False

    # I do not know why from_pretrained will random initialize the Lambada.
    if training_args.do_train and args.quant_method in ["ultraquantv3", "ultraquantv4"]:
        for name, param in model.named_parameters():
            if "Lambada" in name:
                param.data.copy_(
                    torch.randn_like(param) * training_args.range_of_lambada
                )
                # param.requires_grad = True
                # param.data.copy_(torch.zeros_like(param))

    if not args.contain_weight_clip_val:
        for name, param in model.named_parameters():
            if "weight_clip_val" in name:
                weight_name = name.replace("weight_clip_val", "weight")
                weight_param = dict(model.named_parameters()).get(weight_name, None)

                if args.w_bits == 1:
                    scale = torch.mean(
                        weight_param.abs(), dim=-1, keepdim=True
                    ).detach()
                elif args.w_bits == 0 or args.w_bits == 2:
                    scale, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                elif args.w_bits == 3 or args.w_bits == 4:
                    xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    maxq = 2 ** (args.w_bits - 1) - 1
                    scale = xmax / maxq
                else:
                    raise NotImplementedError

                param.data.copy_(scale)
    # # I do not know why from_pretrained will random initialize the Lambada.
    # if training_args.do_train:
    #     for name, param in model.named_parameters():
    #         if "Lambada" in name:
    #             param.data.copy_(torch.randn_like(param)*training_args.range_of_lambada)
    #             # param.data.copy_(torch.zeros_like(param))

    model.cuda()
    log.info("Complete model loading...")

    log.info("Start to load tokenizer...")
    tokenizer = transformers.LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        add_bos_token=False,
        add_eos_token=False,
    )

    DEFAULT_PAD_TOKEN = "[PAD]"
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    log.info("Complete tokenizer loading...")

    model.config.use_cache = False

    if training_args.do_train:
        data_module = make_data_module(tokenizer, args)
        train_data, valid_data, data_collator = (
            data_module["train_dataset"],
            data_module["eval_dataset"],
            data_module["data_collator"],
        )
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_data if training_args.do_train else None,
            eval_dataset=valid_data if training_args.do_eval else None,
            data_collator=data_collator,
        )
        _ = trainer.train()
        trainer.save_state()
        utils.safe_save_model_for_hf_trainer(trainer, args.output_model_local_path)

    # Evaluation
    if training_args.do_eval:
        # model.to("cuda")
        # metrics = trainer.evaluate()
        # max_eval_samples = len(valid_data)
        # metrics["eval_samples"] = min(max_eval_samples, len(valid_data))
        # try:
        #     perplexity = math.exp(metrics["eval_loss"])
        # except OverflowError:
        #     perplexity = float("inf")
        # metrics["perplexity"] = perplexity

        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)
        if args.eval_tasks != "" or args.do_mmlu_eval:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            from lm_eval.tasks import TaskManager
            from lm_eval.utils import make_table

        if args.eval_tasks != "":
            task_list = args.eval_tasks.split(",")
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
                total_acc += results["results"][task]["acc,none"]
                if args.report_to == "wandb":
                    wandb.log(
                        {f"eval/{task}_acc": results["results"][task]["acc,none"]}
                    )
            log.info(f"Average Acc: {total_acc / len(task_list) * 100:.2f}%")

        if args.do_mmlu_eval:
            lm_eval_model = HFLM(pretrained=model, batch_size=16)
            task_manager = TaskManager()
            results = lm_eval.simple_evaluate(  # call simple_evaluate
                model=lm_eval_model,
                tasks=["mmlu"],
                num_fewshot=5,
                task_manager=task_manager,
                cache_requests=True,
            )
            log.info(make_table(results))
            total_acc = 0
            for task in results["results"]:
                total_acc += results["results"][task]["acc,none"]
                if args.report_to == "wandb":
                    wandb.log(
                        {f"eval/{task}_acc": results["results"][task]["acc,none"]}
                    )
            log.info(
                f"Average MMLU Acc: {total_acc / len(results['results']) * 100:.2f}%"
            )

    torch.distributed.barrier()


if __name__ == "__main__":
    train()
