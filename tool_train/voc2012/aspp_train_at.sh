#!/bin/sh
PYTHON=/home/aivs/anaconda3/envs/adv_train/bin/python

dataset=voc2012
source_layer=layer3_2
attack=pgd
at_iter=3
exp_name=aspp_at
exp_dir=exp/${dataset}/${exp_name}_${attack}${at_iter}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool_train/voc2012/aspp_train_at.sh tool_train/train_at_aspp.py ${config} ${exp_dir}

export PYTHONPATH=./
$PYTHON -u tool_train/train_at_aspp.py \
  --config=${config}  --attack=${attack} --at_iter=${at_iter} --source_layer=${source_layer}\
  2>&1 | tee ${model_dir}/train-$now.log

