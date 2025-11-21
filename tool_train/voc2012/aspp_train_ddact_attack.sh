#!/bin/sh
PYTHON=/home/aivs-13/anaconda3/envs/adv_at/bin/python

dataset=voc2012
attack=fspgd
exp_name=aspp_ddcat
exp_dir=exp/${dataset}/${exp_name}/${attack}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_attack.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool_train/voc2012/aspp_train_ddcat_attack.sh tool_train/train_ddcat_aspp_attack.py ${config} ${exp_dir}

export PYTHONPATH=./
#srun -p $PARTITION -n1 --gres=gpu:8 --ntasks-per-node=8 --job-name=python \
$PYTHON -u tool_train/train_ddcat_aspp_attack.py \
  --config=${config} \
  2>&1 | tee ${model_dir}/train-$now.log


#sh tool_train/voc2012/aspp_train_ddcat.sh