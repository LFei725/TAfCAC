import os
import sys
from run_classifier import main

TASK_NAME="ocnli"
MODEL_NAME="./chinese_roberta_wwm_ext_L-12_H-768_A-12"

CURRENT_DIR= "../classifier_pytorch"

CUDA_VISIBLE_DEVICES="0"
BERT_PRETRAINED_MODELS_DIR=CURRENT_DIR+"/prev_trained_model"
BERT_WWM_DIR=BERT_PRETRAINED_MODELS_DIR+"/"+MODEL_NAME
GLUE_DATA_DIR=CURRENT_DIR+"/CLUEdatasets"

if not os.path.exists(GLUE_DATA_DIR):
    os.mkdir(GLUE_DATA_DIR)
    print("makedir " + GLUE_DATA_DIR)


print("****Start Running****")
"""----------运行 run_classifier.py---微调全部层------"""
"""--------per_gpu_train_batch_size最初=16  epoch也需要调整
OCNLI
--data_dir="./CLUEdatasets/ocnli/"
ocnli-public
--data_dir="./CLUEdatasets/ocnli-public/"
*****************
--model_type=bert   
--model_name_or_path="/mnt/pretrained_language_models/bert-base-chinese/"
--data_dir="./CLUEdatasets/ocnli-public/"
--output_dir="./outputs/ocnli_output/"
--per_gpu_train_batch_size=16 
--per_gpu_eval_batch_size=16 
--num_train_epochs=10.0
--learning_rate=3e-5 
--task_name="ocnli"
--cache_dir="./cache_dir/"
--do_train 
--do_eval 
--do_lower_case 
--max_seq_length=512
--logging_steps=24487 
--save_steps=24487 
--overwrite_output_dir 
--seed=42
***************
--------per_gpu_train_batch_size最初=16  epoch也需要调整
python run_classifier.py \
--model_type=bert \
--model_name_or_path="/mnt/pretrained_language_models/bert-base-chinese/" \
--data_dir="./CLUEdatasets/ocnli-public/" \
--output_dir="./outputs/ocnli_output/" \
--per_gpu_train_batch_size=16 \
--per_gpu_eval_batch_size=16 \
--num_train_epochs=10.0 \
--learning_rate=3e-5 \
--task_name="ocnli" \
--cache_dir="./cache_dir/" \
--do_train \
--do_eval \
--do_lower_case \
--max_seq_length=512 \
--logging_steps=24487 \
--save_steps=24487 \
--overwrite_output_dir \
--seed=42
"""

"""---------如果是要预测----------------"""
"""
--model_type=bert 
--model_name_or_path="/mnt/pretrained_language_models/bert-base-chinese/"
--task_name="ocnli"
--do_predict 
--do_lower_case 
--data_dir="./CLUEdatasets/ocnli/"
--max_seq_length=128 
--per_gpu_train_batch_size=16 
--per_gpu_eval_batch_size=16 
--learning_rate=3e-5 
--num_train_epochs=2.0 
--logging_steps=24487 
--save_steps=24487 
--output_dir="./outputs/ocnli_output/"
--overwrite_output_dir 
--seed=42
--------------------------------------"""

"""---------如果是要调试-----------------"""
"""
--model_type=bert   
--model_name_or_path="/mnt/pretrained_language_models/bert-base-chinese/"
--task_name="ocnli"
--cache_dir="./cache_dir/"
--do_train 
--do_eval 
--do_lower_case 
--data_dir="./CLUEdatasets/ocnli/"
--max_seq_length=128 
--per_gpu_train_batch_size=32  
--per_gpu_eval_batch_size=16 
--learning_rate=3e-5 
--num_train_epochs=2.0
--logging_steps=24487 
--save_steps=24487 
--output_dir="./outputs/ocnli_output/"
--overwrite_output_dir 
--seed=42
"""
