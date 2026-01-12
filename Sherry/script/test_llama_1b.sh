mdp=/path/to/trained_model/ 

python3 test_llama.py \
--model_path $mdp \
--w_bits 0 \
--eps 1e-3 \
--quant_method "sherry" \
--granularity "per_group" \
--model_family "llama" \
--group_size 128 \
--do_train False \
--do_eval True \
--N 3 \
--M 4 \
--model_max_length 1024 \
--pt_context_len 1024 \
--source_max_len 1024 \
--target_max_len 1024 \
--fp16 False \
--bf16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--report_to "none" \
--contain_weight_clip_val False \
--do_mmlu_eval False \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \