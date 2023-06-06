#!/usr/bin/env bash
set -e 

dataset_name='imdb-binary'
dataset_init='ones'
dataset_eval='rocauc'
model='GIN_IMDB'
device=2
nlayer=1
embed_dim=128
norm_type='motifnorm'
activation='relu'
dropout=0.0
pool_type='mean'
epochs=300
batch_size=32
lr=1e-3
lr_min=1e-5
lr_patience=10
wd=0.0
seed=0

for lr in 1e-3;do
for norm_type in 'motifnorm';do
for dataset_eval in 'rocauc' 'acc';do
for seed in $(seq 0 9);do

    python main_imdb.py \
        --dataset_name $dataset_name \
        --dataset_init $dataset_init \
        --dataset_eval $dataset_eval \
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
        --lr_warmup  

done
done
done
done

