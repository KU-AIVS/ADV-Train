#sh script/psp2aspp_clean_train_test.sh

#!/bin/sh
PYTHON=/home/aivs/anaconda3/envs/adv_t/bin/python

dataset=voc2012
source_layer=layer3_2
exp_name=pspnet

num=1


exp_dir=exp/${dataset}/${exp_name}/${num}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p ${model_dir} ${result_dir}
export PYTHONPATH=./
num_epoch=50

#for num in `seq 1 5` # average num
#do

# ------------------------TRAIN------------------------
#cp script/psp_clean_train_test.sh tool_train/train_psp.py ${config} ${exp_dir}
#$PYTHON -u tool_train/train_psp.py \
#  --config=${config} --train_num=${num}\
#  2>&1 | tee ${model_dir}/train-$now.log

# ------------------------TEST------------------------
cp tool_test/voc2012/test_voc_psp2aspp.py ${exp_dir}
##PGD1
#$PYTHON -u tool_test/voc2012/test_voc_psp2aspp.py \
#  --config=${config} --test_attack pgd --attack_iter=1 --num_epoch=${num_epoch} --train_num=${num}\
#  2>&1 | tee -a ${result_dir}/test-${num}.log
##SegPGD1
#$PYTHON -u tool_test/voc2012/test_voc_psp2aspp.py \
#  --config=${config} --test_attack segpgd --attack_iter=1 --num_epoch=${num_epoch} --train_num=${num}\
#  2>&1 | tee -a ${result_dir}/test-${num}.log
##CosPGD1
#$PYTHON -u tool_test/voc2012/test_voc_psp2aspp.py \
#  --config=${config} --test_attack cospgd --attack_iter=1 --num_epoch=${num_epoch} --train_num=${num}\
#  2>&1 | tee -a ${result_dir}/test-${num}.log
##fs_yg1
#$PYTHON -u tool_test/voc2012/test_voc_psp2aspp.py \
#  --config=${config} --test_attack fs_yg --attack_iter=1 --num_epoch=${num_epoch} --train_num=${num} --source_layer=${source_layer}\
#  2>&1 | tee -a ${result_dir}/test-${num}.log
#
#
#$PYTHON -u tool_test/result_print.py \
#  --config=${config} --num_epoch=${num_epoch} --train_num=${num}\
#  2>&1 | tee -a ${result_dir}/test-${num}.log
#
##done
#
## ------------------------AVERAGE RESULT------------------------
#$PYTHON -u tool_test/result_print.py \
#  --config=${config} --num_epoch=${num_epoch} --avg=5\
#  2>&1 | tee ${exp_dir}/test-adverage.log