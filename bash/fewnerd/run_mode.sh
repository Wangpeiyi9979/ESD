#!/usr/bin/env bash
N=$3
K=$4
seeds=(1 2 3 4 5)
modes=$2
bert_lr=2e-5
gard_iter=2
gpu=$1
lr=5e-4
droupout=0.1
max_o_num=1000
# shellcheck disable=SC2068
for k in ${K[@]};do
  for mode in ${modes[@]};do
    for s in ${seeds[@]};do
        CUDA_VISIBLE_DEVICES=${gpu} python3 train_demo.py  \
        --lr ${lr} \
        --N ${N} \
        --K ${k} \
        --mode ${mode} \
        --dataset fewnerd \
        --warmup_step 1000 \
        --seed ${s} \
        --num_heads 1 \
        --bert_lr ${bert_lr} \
        --dropout ${droupout} \
        --grad_iter ${gard_iter} \
        --bert_path bert-base-uncased \
        --max_o_num ${max_o_num} \
        --hidsize 100 
    done
  done
  done

