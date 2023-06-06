#!/usr/bin/env bash
set -e 

dataset_name='ogbg-moltoxcast'
model='GCN'
device=2
nlayer=4
embed_dim=128
norm_type='nodenorm'
activation='relu'
dropout=0.5
pool_type='mean'
epochs=450
batch_size=128
lr=1e-3
lr_min=1e-5
lr_patience=15
wd=0.0
seed=0

for nlayer in 4 16 32;do
for lr in 1e-3;do
for wd in 0.0;do
for seed in 0 1 2 3 4 5 6 7 8 9;do

    python main_graph.py \
        --dataset_name $dataset_name \
        --model $model \
        --device $device \
        --num_layer $nlayer \
        --embed_dim $embed_dim \
        --norm_type $norm_type \
        --activation $activation \
        --dropout $dropout \
        --pool_type $pool_type \
        --epochs $epochs \
        --batch_size $batch_size \
        --lr $lr \
        --lr_min $lr_min \
        --lr_patience $lr_patience \
        --weight_decay $wd\
        --seed $seed\
        --breakout

done
done
done
done
