#!/usr/bin/env bash

lr=5e-4
dropout=0.4
embedding_dim=64
neighbor_embedding_dim=32
hidden_dims="64 32"
epochs=64
alpha=0.5
gamma=2
lamda=0.8
neighbor=3
seed=666
comment="eval2-2"
dataset="Fdataset"
python demo.py --dataset_name ${dataset} --lr ${lr} --dropout ${dropout} \
   --embedding_dim ${embedding_dim} --neighbor_embedding_dim ${neighbor_embedding_dim} \
   --hidden_dims ${hidden_dims} --epochs ${epochs} --alpha ${alpha} --gamma ${gamma} --lamda ${lamda} \
   --comment ${comment} --seed ${seed} --drug_neighbor_num ${neighbor} --disease_neighbor_num ${neighbor}

