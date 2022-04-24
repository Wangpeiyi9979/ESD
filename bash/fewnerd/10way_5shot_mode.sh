#!/usr/bin/env bash
N=10
K=5
seeds=(1 2 3 4 5)
modes=$2
bert_lr=2e-5
gard_iter=2
gpu=$1
lr=5e-4
droupout=0.1
eposide_tasks=20
max_o_num=120 # down sampling

for mode in ${modes[@]};do
  for s in ${seeds[@]};do
      CUDA_VISIBLE_DEVICES=${gpu} python3 train_demo.py  \
      --lr ${lr} \
      --N ${N} \
      --K ${K} \
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
      --hidsize 100 \
      --eposide_tasks ${eposide_tasks} 
  done
done

