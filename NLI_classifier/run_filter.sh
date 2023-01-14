#!/bin/bash

SRC_PATH="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/new_overgenerate/"
DES_PATH="/mnt/3_new-try/task_augment/codes/now_use/OCNLI-classifier-pytorch/my_data/"
MODEL_PATH="best_OCNLI_classify"
OUTPUT_DIR="ogfilter_output/"
MAX_LEN=512

for file in "train" "test" "fu_train" "chu_test"
do
  for nli_s in "contradiction" "entailment" "neutral"
  do
    cp ${SRC_PATH}NLI_${nli_s}_${file}.json ${DES_PATH}test.json
    echo "copy ${SRC_PATH}NLI_${nli_s}_${file}.json success"
    RIGHT_LABEL=0
    if [ ${nli_s} == "contradiction" ]
    then
      RIGHT_LABEL=0
    elif [ ${nli_s} == "entailment" ]
    then
      RIGHT_LABEL=1
    else
      RIGHT_LABEL=2
    fi
    echo ${RIGHT_LABEL}
    python filter_sentence.py \
      --model_type=bert \
      --model_name_or_path="./outputs/${MODEL_PATH}/" \
      --output_filter_dir_name="./my_data/${OUTPUT_DIR}${file}_${nli_s}.json" \
      --output_dir_add="${file}_${nli_s}/" \
      --now_right_label=0 \
      --output_dir="./my_data/${OUTPUT_DIR}" \
      --task_name="ocnli" \
      --do_predict \
      --do_lower_case \
      --data_dir="./my_data/" \
      --max_seq_length=512 \
      --per_gpu_eval_batch_size=21 \
      --learning_rate=3e-5 \
      --logging_steps=24487 \
      --save_steps=24487 \
      --overwrite_output_dir \
      --seed=42
    rm ${DES_PATH}test.json
    echo "remove success"
    rm ${DES_PATH}cached_test_${MODEL_PATH}_${MAX_LEN}_ocnli
    echo "remove success"
    echo "predict ${SRC_PATH}NLI_${nli_s}_${file}.json over"
    sleep 3s
  done
done




