#!/bin/bash
TS_SOCKET=/tmp/sh0416-gpu0 tsp -C
TS_SOCKET=/tmp/sh0416-gpu1 tsp -C
TS_SOCKET=/tmp/sh0416-gpu2 tsp -C
TS_SOCKET=/tmp/sh0416-gpu3 tsp -C

source /home/sh0416/.bashrc
let gpu=2
seed_list=(0 1 2 3 4)
augment_list=("none" "adamix")
data_augment_list=("backtranslate")

data_dir="/data/sh0416/dataset/pdistmix/ag_news_csv"
dataset="ag_news"
num_train=(2500 10000)
for augment in "${augment_list[@]}"; do
  for data_augment in "${data_augment_list[@]}"; do
    for num_train in "${num_train_list[@]}"; do
      for seed in "${seed_list[@]}"; do
        echo "--data_dir ${data_dir} --dataset ${dataset} --data_augment ${data_augment} --num_train ${num_train} --seed ${seed} --augment ${augment} --gpu ${gpu}"
        TS_SOCKET=/tmp/sh0416-gpu${gpu} tsp python main.py --data_dir ${data_dir} --dataset ${dataset} --data_augment ${data_augment} --num_train ${num_train} --seed ${seed} --augment ${augment} --gpu ${gpu}
        let gpu=$(((gpu+1)%4))
      done
    done
  done
done

data_dir="/data/sh0416/dataset/pdistmix/yahoo_answers_csv"
dataset="yahoo_answer"
num_train_list=(2000 25000)
for augment in "${augment_list[@]}"; do
  for data_augment in "${data_augment_list[@]}"; do
    for num_train in "${num_train_list[@]}"; do
      for seed in "${seed_list[@]}"; do
        echo "--data_dir ${data_dir} --dataset ${dataset} --data_augment ${data_augment} --num_train ${num_train} --seed ${seed} --augment ${augment} --gpu ${gpu}"
        TS_SOCKET=/tmp/sh0416-gpu${gpu} tsp python main.py --data_dir ${data_dir} --dataset ${dataset} --data_augment ${data_augment} --num_train ${num_train} --seed ${seed} --augment ${augment} --gpu ${gpu}
        let gpu=$(((gpu+1)%4))
      done
    done
  done
done

data_dir="/data/sh0416/dataset/pdistmix/amazon_review_polarity_csv"
dataset="amazon_review_polarity"
num_train_list=(2500 10000)
for augment in "${augment_list[@]}"; do
  for data_augment in "${data_augment_list[@]}"; do
    for num_train in "${num_train_list[@]}"; do
      for seed in "${seed_list[@]}"; do
        echo "--data_dir ${data_dir} --dataset ${dataset} --data_augment ${data_augment} --num_train ${num_train} --seed ${seed} --augment ${augment} --gpu ${gpu}"
        TS_SOCKET=/tmp/sh0416-gpu${gpu} tsp python main.py --data_dir ${data_dir} --dataset ${dataset} --data_augment ${data_augment} --num_train ${num_train} --seed ${seed} --augment ${augment} --gpu ${gpu}
        let gpu=$(((gpu+1)%4))
      done
    done
  done
done

data_dir="/data/sh0416/dataset/pdistmix/dbpedia_csv"
dataset="dbpedia"
num_train_list=(2800 35000)
for augment in "${augment_list[@]}"; do
  for data_augment in "${data_augment_list[@]}"; do
    for num_train in "${num_train_list[@]}"; do
      for seed in "${seed_list[@]}"; do
        echo "--data_dir ${data_dir} --dataset ${dataset} --data_augment ${data_augment} --num_train ${num_train} --seed ${seed} --augment ${augment} --gpu ${gpu}"
        TS_SOCKET=/tmp/sh0416-gpu${gpu} tsp python main.py --data_dir ${data_dir} --dataset ${dataset} --data_augment ${data_augment} --num_train ${num_train} --seed ${seed} --augment ${augment} --gpu ${gpu}
        let gpu=$(((gpu+1)%4))
      done
    done
  done
done
