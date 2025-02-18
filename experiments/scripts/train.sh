#!/bin/bash

export PYTHONPATH="`pwd`:${PYTHONPATH}"
if [ $# -lt 4 ]
then
  echo "Please specify 1) cfg; 2) gpus; 3) method; 4) exp_name."
  exit
fi

cfg=${1}
gpus=${2}
method=${3}
exp_name=${4}
shift 4

out_dir=./experiments/ckpt/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}

# CUDA_VISIBLE_DEVICES=${gpus}
# Note we accept additional command-line arguments here
python3 ./tools/train.py --cfg ${cfg} \
  --method ${method} --exp_name ${exp_name} "$@" 2>&1 | tee ${out_dir}/log.txt
