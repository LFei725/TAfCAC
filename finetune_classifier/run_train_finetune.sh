python run_classifier.py \
--model_type=bert \
--model_name_or_path="/mnt/pretrained_language_models/bert-base-chinese/" \
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/ogfilter_output/组合好的用于NLI训练的traintestdev/" \
--output_dir="./outputs/legal_after_filter/" \
--max_seq_length=128 \
--per_gpu_train_batch_size=32 \
--per_gpu_eval_batch_size=16 \
--learning_rate=3e-5 \
--num_train_epochs=20.0 \
--task_name="ocnli" \
--cache_dir="./cache_dir/" \
--do_train \
--do_eval \
--do_lower_case \
--logging_steps=24487 \
--save_steps=24487 \
--overwrite_output_dir \
--seed=42

python run_classifier_1.py \
--model_type=bert \
--model_name_or_path="./outputs/new_filter_legal_output/nli_legal/" \
--output_dir="./outputs/new_filter_legal_output/" \
--data_dir="./CLUEdatasets/ocnli/" \
--max_seq_length=128 \
--per_gpu_train_batch_size=32 \
--per_gpu_eval_batch_size=16 \
--num_train_epochs=20.0 \
--learning_rate=3e-5 \
--task_name="ocnli" \
--cache_dir="./cache_dir/" \
--do_train \
--do_eval \
--do_lower_case \
--logging_steps=24487 \
--save_steps=24487 \
--overwrite_output_dir \
--seed=42

