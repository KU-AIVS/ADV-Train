#sh tool_test/voc2012/psp_test_at.sh


#!/bin/sh
PYTHON=/home/aivs-13/anaconda3/envs/adv_at/bin/python

dataset=voc2012
attack=fspgd
at_iter=3
exp_name=pspnet50_at
exp_dir=exp/${dataset}/${exp_name}_${attack}${at_iter}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
cp tool_test/voc2012/psp_test_at.sh tool_test/voc2012/test_voc_psp_at.py ${config} ${exp_dir}

export PYTHONPATH=./
#srun -p $PARTITION -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=python \

#Clean
$PYTHON -u tool_test/voc2012/test_voc_psp_at.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/test-$now.log
#CW
$PYTHON -u tool_test/voc2012/test_voc_psp_at.py \
  --config=${config} --test_attack cw20 \
  2>&1 | tee ${result_dir}/test-$now.log
#DeepFool(DF)
$PYTHON -u tool_test/voc2012/test_voc_psp_at.py \
  --config=${config} --test_attack df20 \
  2>&1 | tee ${result_dir}/test-$now.log
#BIM
$PYTHON -u tool_test/voc2012/test_voc_psp_at.py \
  --config=${config} --test_attack bim20 \
  2>&1 | tee ${result_dir}/test-$now.log
#PGD10
$PYTHON -u tool_test/voc2012/test_voc_psp_at.py \
  --config=${config} --test_attack pgd10 \
  2>&1 | tee ${result_dir}/test-$now.log
#PGD20
$PYTHON -u tool_test/voc2012/test_voc_psp_at.py \
  --config=${config} --test_attack pgd20 \
  2>&1 | tee ${result_dir}/test-$now.log
#PGD40
$PYTHON -u tool_test/voc2012/test_voc_psp_at.py \
  --config=${config} --test_attack pgd40 \
  2>&1 | tee ${result_dir}/test-$now.log
#PGD100
$PYTHON -u tool_test/voc2012/test_voc_psp_at.py \
  --config=${config} --test_attack pgd100 \
  2>&1 | tee ${result_dir}/test-$now.log