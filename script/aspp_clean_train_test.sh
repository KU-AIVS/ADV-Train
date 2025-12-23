#sh script/aspp_clean_train_test.sh

#!/bin/sh
PYTHON=/home/aivs/anaconda3/envs/adv_training/bin/python
source_layer=layer3_2
dataset=voc2012
exp_name=aspp

num=1
#for num in `seq 1 5` # average num
#do

# ------------------------TRAIN------------------------
exp_dir=exp/${dataset}/${exp_name}/${num}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp script/aspp_clean_train_test.sh tool_train/train_aspp.py ${config} ${exp_dir}
export PYTHONPATH=./
$PYTHON -u tool_train/train_aspp.py \
  --config=${config} --train_num=${num}\
  2>&1 | tee ${model_dir}/train-$now.log

# ------------------------TEST------------------------
num_epoch=50
cp tool_test/voc2012/test_voc_aspp.py ${exp_dir}
#Clean
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee ${result_dir}/test-Clean.log
#---------------PGD------------------------------------
#PGD3
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack pgd --attack_iter=3 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#PGD5
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack pgd --attack_iter=5 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#PGD7
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack pgd --attack_iter=7 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#PGD10
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack pgd --attack_iter=10 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#PGD20
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack pgd --attack_iter=20 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#PGD40
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack pgd --attack_iter=40 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#PGD100
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack pgd --attack_iter=100 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log

#---------------SegPGD------------------------------------
#SegPGD3
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack segpgd --attack_iter=3 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#SegPGD5
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack segpgd --attack_iter=5 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#SegPGD7
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack segpgd --attack_iter=7 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#SegPGD10
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack segpgd --attack_iter=10 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#SegPGD20
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack segpgd --attack_iter=20 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#SegPGD40
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack segpgd --attack_iter=40 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#SegPGD100
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack segpgd --attack_iter=100 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log

#---------------CosPGD------------------------------------
#CosPGD3
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack cospgd --attack_iter=3 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#CosPGD5
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack cospgd --attack_iter=5 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#CosPGD7
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack cospgd --attack_iter=7 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#CosPGD10
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack cospgd --attack_iter=10 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#CosPGD20
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack cospgd --attack_iter=20 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#CosPGD40
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack cospgd --attack_iter=40 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#CosPGD100
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack cospgd --attack_iter=100 --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log

#---------------fs_yg------------------------------------
#fs_yg3
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack fs_yg --attack_iter=3 --num_epoch=${num_epoch} --train_num=${num} --source_layer=${source_layer}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#fs_yg5
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack fs_yg --attack_iter=5 --num_epoch=${num_epoch} --train_num=${num} --source_layer=${source_layer}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#fs_yg7
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack fs_yg --attack_iter=7 --num_epoch=${num_epoch} --train_num=${num} --source_layer=${source_layer}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#fs_yg10
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack fs_yg --attack_iter=10 --num_epoch=${num_epoch} --train_num=${num} --source_layer=${source_layer}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#fs_yg20
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack fs_yg --attack_iter=20 --num_epoch=${num_epoch} --train_num=${num} --source_layer=${source_layer}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#fs_yg40
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack fs_yg --attack_iter=40 --num_epoch=${num_epoch} --train_num=${num} --source_layer=${source_layer}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#fs_yg100
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack fs_yg --attack_iter=100 --num_epoch=${num_epoch} --train_num=${num} --source_layer=${source_layer}\
  2>&1 | tee -a ${result_dir}/test-${num}.log

$PYTHON -u tool_test/result_print.py \
  --config=${config} --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log

#done

# ------------------------AVERAGE RESULT------------------------
$PYTHON -u tool_test/result_print.py \
  --config=${config} --num_epoch=${num_epoch} --avg=5\
  2>&1 | tee ${exp_dir}/test-adverage.log