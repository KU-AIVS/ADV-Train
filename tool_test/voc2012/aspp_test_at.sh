#sh tool_test/voc2012/aspp_test_at.sh


#!/bin/sh
PYTHON=/home/aivs/anaconda3/envs/adv_train/bin/python

dataset=voc2012
attack=pgd
at_iter=3
source_layer=layer3_2
exp_name=aspp_at
exp_dir=exp/${dataset}/${exp_name}_${attack}${at_iter}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
cp tool_test/voc2012/aspp_test_at.sh tool_test/voc2012/test_voc_aspp_at.py ${config} ${exp_dir}

export PYTHONPATH=./
#srun -p $PARTITION -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=python \

#Clean
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --attack=${attack} --at_iter=${at_iter} --source_layer=${source_layer}\
  2>&1 | tee ${result_dir}/test-$now.log
#CW
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --test_attack cw20  --attack=${attack} --at_iter=${at_iter} --source_layer=${source_layer}\
  2>&1 | tee ${result_dir}/test-$now.log
#DeepFool(DF)
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --test_attack df20  --attack=${attack} --at_iter=${at_iter} --source_layer=${source_layer}\
  2>&1 | tee ${result_dir}/test-$now.log
#BIM
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --test_attack bim20  --attack=${attack} --at_iter=${at_iter} --source_layer=${source_layer}\
  2>&1 | tee ${result_dir}/test-$now.log
#PGD10
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --test_attack pgd10  --attack=${attack} --at_iter=${at_iter} --source_layer=${source_layer}\
  2>&1 | tee ${result_dir}/test-$now.log
#PGD20
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --test_attack pgd20  --attack=${attack} --at_iter=${at_iter} --source_layer=${source_layer}\
  2>&1 | tee ${result_dir}/test-$now.log
#PGD40
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --test_attack pgd40  --attack=${attack} --at_iter=${at_iter} --source_layer=${source_layer}\
  2>&1 | tee ${result_dir}/test-$now.log
#PGD100
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --test_attack pgd100  --attack=${attack} --at_iter=${at_iter} --source_layer=${source_layer}\
  2>&1 | tee ${result_dir}/test-$now.log

$PYTHON -u tool_test/result_print.py \
  --config=${config} --attack=${attack} --at_iter=${at_iter} --source_layer=${source_layer}\
  2>&1 | tee ${result_dir}/test-$now.log