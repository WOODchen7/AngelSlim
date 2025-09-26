#!/bin/bash

NET_TYPE="high"
RESTART_CONFIG="false"
CHECK_INTERVAL=1200
export NCCL_IB_TIMEOUT=24
export NCCL_PROFILE_PRIMS=0
export NCCL_PROFILE_PRIMS_ENABLE=0
export NCCL_NVLS_ENABLE=0
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_TOPO_AFFINITY=0
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=0
fi

export HOST_GPU_NUM=8
# 当前机器ip
export LOCAL_IP=$LOCAL_IP
# 多节点机器ip，逗号隔开
#export NODE_IP_LIST="${LOCAL_IP}:8"
# 机器节点个数
export NODES=4
export NODE_NUM=$((${NODES} * ${HOST_GPU_NUM}))

export NCCL_DEBUG=WARN

tokenizer_path=${model_path}


echo "NODE_IP_LIST: $NODE_IP_LIST"
echo $NODE_IP_LIST > env.txt 2>&1 &
sed "s/:/ slots=/g" env.txt | sed "s/,/\n/g" >  "hostfile"
sed "s/:.//g" env.txt | sed "s/,/\n/g" >  "pssh.hosts"
export CHIEF_IP=$LOCAL_IP

HOST_PATH=hostfile

echo "###CHECK###"
cat $HOST_PATH
echo $CHIEF_IP


deepspeed --hostfile=$HOST_PATH --master_addr $CHIEF_IP train.py \
--dataset_root "dataset_root" \
--local_dir "/BitQ_output/" \
--model_path "/Llama-3.2-1B/" \
--model_family "llama-3.2-1B" \
--dataset "ultrafineweb-10b" \
--dataset_format "pt" \
--learning_rate 1e-3 \
--weight_decay 0. \
--warmup_ratio 0.61 \
--lr_scheduler_type "warmup_stable_decay" \
--my_lr_scheduler_kwargs '{"min_lr_ratio" : 1e-1, "num_decay_steps" : 14592, "num_stable_steps" : 0}' \
--output_model_filename "1B-1epoch-lr1e-4-uqv2_lrud-eps1e-3" \
--quant_method "ultraquantv2" \
--eps 1e-3 \
--granularity "per_group" \
--group_size 128 \
--num_train_epochs 1.0 \
--w_bits 0 \
--model_max_length 1024 \
--pt_context_len 1024 \
--source_max_len 1024 \
--target_max_len 1024 \
--do_train True \
--do_eval False \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir "./output/Llama-3.2-1B" \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--save_strategy "steps" \
--save_steps 1500 \
--report_to "none" \
--save_total_limit 40 \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing False \
--qat True \
--do_mmlu_eval True \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \



