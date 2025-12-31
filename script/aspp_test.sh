#sh script/aspp_clean_train_test.sh

#!/bin/sh
PYTHON=/home/aivs/anaconda3/envs/adv_training/bin/python
source_layer=layer3_2
dataset=voc2012
exp_name=aspp
fusion=False
if [ "$fusion" = "True" ]; then
    fusion_cmd="--fusion"
else
    fusion_cmd=""
fi
num=1
# ------------------------TRAIN------------------------
exp_dir=exp/${dataset}/${exp_name}/${num}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")
export PYTHONPATH=./
mkdir -p ${model_dir} ${result_dir}

# ------------------------TEST------------------------
num_epoch=50
cp tool_test/voc2012/test_voc_aspp.py ${exp_dir}
#------Clean
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee ${result_dir}/test-Clean.log
#-----Attack
#1 3 5 7 10 20

for attack_iter in 2
do
#PGD
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack pgd --attack_iter=${attack_iter} --num_epoch=${num_epoch} --train_num=${num} ${fusion_cmd}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#SegPGD
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack segpgd --attack_iter=${attack_iter} --num_epoch=${num_epoch} --train_num=${num} ${fusion_cmd}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#CosPGD
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack cospgd --attack_iter=${attack_iter} --num_epoch=${num_epoch} --train_num=${num} ${fusion_cmd}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
#fs_yg
$PYTHON -u tool_test/voc2012/test_voc_aspp.py \
  --config=${config} --test_attack fs_yg --attack_iter=${attack_iter} --num_epoch=${num_epoch} --train_num=${num} ${fusion_cmd}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
done
$PYTHON -u tool_test/result_print.py \
  --config=${config} --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
