#!/usr/bin/env bash
seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
mode=7
bert_lr=5e-5
gard_iter=1
gpu=$1
lr=5e-4
droupout=0.1
for s in ${seeds[@]};do
    CUDA_VISIBLE_DEVICES=${gpu} python3 train_demo.py  \
    --lr ${lr} \
    --K 1 \
    --mode ${mode} \
    --dataset snips \
    --seed ${s} \
    --bert_lr ${bert_lr} \
    --dropout ${droupout} \
    --grad_iter ${gard_iter} \
    --bert_path bert-base-uncased \
    --warmup_step 1000 \
    --num_heads 1 \
    --max_o_num 1000 \
    --early_stop 6 \
    --hidsize 100 \
    --eposide_tasks 20 \
    --batch_size 4 \
    --shuffle \
    --max_epoch 10 
done


