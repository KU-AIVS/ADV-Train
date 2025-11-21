#sh tool_test/voc2012/aspp_test_ddcat.sh


#!/bin/sh
PYTHON=/home/aivs/anaconda3/envs/adv_train/bin/python

dataset=voc2012
exp_name=aspp_ddcat
exp_dir=exp/${dataset}/${exp_name}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
cp tool_test/voc2012/aspp_test_ddcat.sh tool_test/voc2012/test_voc_aspp_ddcat.py ${config} ${exp_dir}

export PYTHONPATH=./
#srun -p $PARTITION -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=python \

##Clean
#$PYTHON -u tool_test/voc2012/test_voc_aspp_ddcat.py \
#  --config=${config} \
#  2>&1 | tee ${result_dir}/test-$now.log

##CW
#$PYTHON -u tool_test/voc2012/test_voc_aspp_ddcat.py \
#  --config=${config} --test_attack cw20 \
#  2>&1 | tee ${result_dir}/test-$now.log
##DeepFool(DF)
#$PYTHON -u tool_test/voc2012/test_voc_aspp_ddcat.py \
#  --config=${config} --test_attack df20\
#  2>&1 | tee ${result_dir}/test-$now.log
##BIM
#$PYTHON -u tool_test/voc2012/test_voc_aspp_ddcat.py \
#  --config=${config} --test_attack bim20 \
#  2>&1 | tee ${result_dir}/test-$now.log
##PGD10
#$PYTHON -u tool_test/voc2012/test_voc_aspp_ddcat.py \
#  --config=${config} --test_attack pgd10 \
#  2>&1 | tee ${result_dir}/test-$now.log
##PGD20
#$PYTHON -u tool_test/voc2012/test_voc_aspp_ddcat.py \
#  --config=${config} --test_attack pgd20 \
#  2>&1 | tee ${result_dir}/test-$now.log
##PGD40
#$PYTHON -u tool_test/voc2012/test_voc_aspp_ddcat.py \
#  --config=${config} --test_attack pgd40 \
#  2>&1 | tee ${result_dir}/test-$now.log
##PGD100
#$PYTHON -u tool_test/voc2012/test_voc_aspp_ddcat.py \
#  --config=${config} --test_attack pgd100 \
#  2>&1 | tee ${result_dir}/test-$now.log



$PYTHON -u tool_test/result_print_ddcat.py \
  --config=${config}\
  2>&1 | tee ${result_dir}/test-$now.log