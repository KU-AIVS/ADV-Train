#sh script/aspp_fusion_at.sh

#!/bin/sh
PYTHON=/home/aivs/anaconda3/envs/adv_attack/bin/python

# ------------------------Config Setting------------------------
dataset=voc2012
exp_name=aspp_fusion_at
num=1

config=config/${dataset}/${dataset}_${exp_name}.yaml
exp_dir=exp/${dataset}/${exp_name}/${num}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
mkdir -p ${model_dir} ${result_dir}
cp script/aspp_fusion_at.sh ${exp_dir}
now=$(date +"%Y%m%d_%H%M%S")
export PYTHONPATH=./
num_epoch=50

# ------------------------AT Startegy------------------------
source_layer=layer3_2
attack=pgd
at_iter=3
fusion=True
if [ "$fusion" = "True" ]; then
    fusion_cmd="--fusion"
else
    fusion_cmd=""
fi
# -----------------------------------------------------------

# ------------------------TRAIN------------------------
cp tool_train/train_fusion_at_aspp.py ${config} ${exp_dir}
$PYTHON -u tool_train/train_fusion_at_aspp.py \
  --config=${config} --attack=${attack} --at_iter=${at_iter} --source_layer=${source_layer} --train_num=${num}  ${fusion_cmd}\
  2>&1 | tee ${model_dir}/train-$now.log
# -----------------------------------------------------

#
# ------------------------TEST------------------------
cp tool_test/voc2012/test_voc_aspp_at.py ${exp_dir}

# ------Clean
$PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
  --config=${config} --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee ${result_dir}/test-Clean.log

#-----Attack
for attack_iter in 10 20 40 100
do
  #PGD
  $PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
    --config=${config} --test_attack pgd --attack_iter=${attack_iter} --num_epoch=${num_epoch} --train_num=${num}\
    2>&1 | tee -a ${result_dir}/test-${num}.log
  #SegPGD
  $PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
    --config=${config} --test_attack segpgd --attack_iter=${attack_iter} --num_epoch=${num_epoch} --train_num=${num}\
    2>&1 | tee -a ${result_dir}/test-${num}.log
  #CosPGD
  $PYTHON -u tool_test/voc2012/test_voc_aspp_at.py \
    --config=${config} --test_attack cospgd --attack_iter=${attack_iter} --num_epoch=${num_epoch} --train_num=${num}\
    2>&1 | tee -a ${result_dir}/test-${num}.log
done

$PYTHON -u tool_test/result_print.py \
  --config=${config} --num_epoch=${num_epoch} --train_num=${num}\
  2>&1 | tee -a ${result_dir}/test-${num}.log
