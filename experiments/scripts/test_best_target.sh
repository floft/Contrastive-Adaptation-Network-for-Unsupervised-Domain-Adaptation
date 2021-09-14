#!/bin/bash

export PYTHONPATH="`pwd`:${PYTHONPATH}"
if [ $# != 6 ]
then
  echo "Please specify 1) cfg; 2) gpus; 3) if adapted; 4) train_exp_name; 5) exp_name.; 6) savedir"
  exit
fi

cfg=${1}
gpus=${2}
adapted=${3}
train_exp_name=${4}
exp_name=${5}
savedir=${6}  # probably: ./experiments/ckpt

# To be comparable with CoDATS, we want to get the best-on-target, but only
# out of 9 models (not 100). So, sort by timestamp and get every nth such that
# we have only 9 total and one of those is the final model (so we do it in
# reverse and always include the 0th, i.e. last, in the tested models).
weights=($(ls -t $savedir/${train_exp_name}/ckpt_*.weights))
# Don't break if there's less than 9 weights
(( ${#weights[@]} >= 9 )) && n=$((${#weights[@]} / 9)) || n=1
logs=()

# for weight in "${weights[@]}"; do
for ((i=0; i<${#weights[@]}-n; i+=n)); do
  weight="${weights[$i+1]}"  # bash is 1-indexed

  iter=$(basename "$weight" | sed -r 's/ckpt_(.*)_(.*)/\1/g')
  out_dir=$savedir/${exp_name}_${iter}
  if [ -d ${out_dir} ]
  then
    rm -rf ${out_dir}
  fi
  mkdir -p ${out_dir}

  if [ x${adapted} = x"True" ]
  then
    # CUDA_VISIBLE_DEVICES=${gpus}
    python3 ./tools/test.py --cfg ${cfg} --adapted \
      --exp_name ${exp_name}_${iter} --weights ${weight} 2>&1 | tee ${out_dir}/log_${iter}.txt
  else
    # CUDA_VISIBLE_DEVICES=${gpus}
    python3 ./tools/test.py --cfg ${cfg} \
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
