#sh script/aspp_adv_train_test.sh

#!/bin/sh
PYTHON=/home/aivs/anaconda3/envs/adv_t/bin/python

dataset=voc2012
source_layer=layer3_2
attack=fs_yg
at_iter=1
exp_name=aspp_at
num=1
#for num in `seq 1 5` # average num
#do

# ------------------------TRAIN------------------------
exp_dir=exp/${dataset}/${exp_name}_${attack}${at_iter}/${num}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp script/aspp_adv_train_test.sh tool_train/train_at_aspp.py ${config} ${exp_dir}
export PYTHONPATH=./
$PYTHON -u tool_train/train_at_aspp.py \
  --config=${config}  --attack=${attack} --at_iter=${at_iter} --source_layer=${source_layer} --train_num=${num}\
  2>&1 | tee ${model_dir}/train-$now.log

# ------------------------TEST------------------------
num_epoch=50
cp tool_test/voc2012/test_voc_aspp_at.py ${exp_dir}
for attack_iter in 1 3 5 7 10 20
do
#PGD
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --attack=${attack} --at_iter=${at_iter} --test_attack pgd --attack_iter=${attack_iter} --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#SegPGD
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --attack=${attack} --at_iter=${at_iter} --test_attack segpgd --attack_iter=${attack_iter} --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#CosPGD
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --attack=${attack} --at_iter=${at_iter} --test_attack cospgd --attack_iter=${attack_iter} --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#fs_yg
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --attack=${attack} --at_iter=${at_iter} --test_attack fs_yg --attack_iter=${attack_iter} --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
done

$PYTHON -u tool_test/result_print.py \
  --config=${config} --attack=${attack} --at_iter=${at_iter} --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#done

# ------------------------AVERAGE RESULT------------------------
$PYTHON -u tool_test/result_print.py \
  --config=${config} --attack=${attack} --at_iter=${at_iter} --num_epoch=${num_epoch} --avg=5\
  2>&1 | tee ${exp_dir}/test-adverage.log