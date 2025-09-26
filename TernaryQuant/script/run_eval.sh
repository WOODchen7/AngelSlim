
mdp="MODEL-PATH"
echo $mdp


python3 ./train_standalone.py \
--model_path $mdp \
--w_bits 0 \
--quant_method "ultraquantv2" \
--eps 1e-3 \
--granularity "per_group" \
--group_size 128 \
--do_train False \
--do_eval True \
--model_max_length 512 \
--pt_context_len 512 \
--source_max_len 512 \
--target_max_len 512 \
--fp16 False \
--bf16 True \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--report_to "none" \
--contain_weight_clip_val True \
--do_mmlu_eval False \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
# --evaluation_strategy "no" \


