#!/bin/bash

export PYTHONPATH="`pwd`:${PYTHONPATH}"
if [ $# != 5 ]
then
  echo "Please specify 1) cfg; 2) gpus; 3) if adapted; 4) train_exp_name; 5) exp_name."
  exit
fi

cfg=${1}
gpus=${2}
adapted=${3}
train_exp_name=${4}
exp_name=${5}

weights=(experiments/ckpt/${train_exp_name}/ckpt_*.weights)
logs=()

for weight in "${weights[@]}"; do
  iter=$(basename "$weight" | sed -r 's/ckpt_(.*)_(.*)/\1/g')
  out_dir=./experiments/ckpt/${exp_name}_${iter}
  if [ -d ${out_dir} ]
  then
    rm -rf ${out_dir}
  fi
  mkdir -p ${out_dir}

  if [ x${adapted} = x"True" ]
  then
    CUDA_VISIBLE_DEVICES=${gpus} python ./tools/test.py --cfg ${cfg} --adapted \
                --exp_name ${exp_name}_${iter} --weights ${weight} 2>&1 | tee ${out_dir}/log_${iter}.txt
  else
    CUDA_VISIBLE_DEVICES=${gpus} python ./tools/test.py --cfg ${cfg} \
                --exp_name ${exp_name}_${iter} --weights ${weight} 2>&1 | tee ${out_dir}/log_${iter}.txt

  fi

  logs+=(${out_dir}/log_${iter}.txt)
done

# Get best one
best_accuracy=0
best_log=""

for log in "${logs[@]}"; do
  accuracy="$(grep "Test mean_accu:" "$log" | cut -d' ' -f3)"

  if awk "BEGIN {exit !($accuracy >= $best_accuracy)}"; then
    best_accuracy=$accuracy
    best_log=$log
  fi
done

echo "The best result on target domain: $log"
echo "Test mean_accu: $best_accuracy"
