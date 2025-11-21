#!/bin/sh
PYTHON=/home/aivs-13/anaconda3/envs/adv_at/bin/python

dataset=voc2012
attack=fspgd
at_iter=3
exp_name=pspnet50_at
exp_dir=exp/${dataset}/${exp_name}/${attack}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool_train/voc2012/psp_train_at.sh tool_train/train_at_psp.py ${config} ${exp_dir}

export PYTHONPATH=./
#srun -p $PARTITION -n1 --gres=gpu:8 --ntasks-per-node=8 --job-name=python \
$PYTHON -u tool_train/train_at_psp.py \
  --config=${config} \
  2>&1 | tee ${model_dir}/train-$now.log
