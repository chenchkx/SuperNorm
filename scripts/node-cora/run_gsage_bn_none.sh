#!/usr/bin/env bash
set -e 

dataset_name='cora'
model='GraphSage'
device=1
nlayer=4
embed_dim=128
norm_type='batchnorm'
activation='relu'
dropout=0.0
epochs=450
lr=1e-2
lr_min=1e-6
lr_patience=10
wd=0.0
seed=0
skip_type='None'

# for norm_type in 'batchnorm' 'None';do
for norm_type in 'batchnorm' 'None';do
for dropout in 0.5;do
for nlayer in $(seq 0 2 42);do
for seed in 0 1 2 3 4 5 6 7 8 9;do


    python main_node.py \
        --dataset_name $dataset_name \
        --model $model \
        --device $device \
        --num_layer $nlayer \
        --embed_dim $embed_dim \
        --norm_type $norm_type \
        --activation $activation \
        --dropout $dropout \
        --skip_type $skip_type \
        --epochs $epochs \
        --lr $lr \
        --lr_min $lr_min \
        --lr_patience $lr_patience \
        --weight_decay $wd\
        --seed $seed\
        --breakout\
        --state_dict

done
done
done
done
