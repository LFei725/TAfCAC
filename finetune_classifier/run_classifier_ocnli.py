"""----------如果是要用legal_NLI训练-----------"""
"""----------运行 run_classifier.py---------"""
"""
****贪婪生成：
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/1648+400+1500+500/组合好的用于NLI训练的traintestdev/"
--output_dir="./outputs/legal_output/"
****过度生成&过滤：
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/ogfilter_output/组合完毕之后的用于NLI训练的traintestdev/"
--output_dir="./outputs/legal_output/"
****使用 微调了第11，pooler，classifier层的NLI分类器 过滤后的数据集
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/ogfilter_output_ft11pc/组合好的用于NLI训练的traintestdev/"
--output_dir="./outputs/legal_after_filter/"
****public+base
****public+base+贪婪生成
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/new_generate/组合好的用于NLI训练的traintestdev/"
--output_dir="./outputs/new_legal_output/"
****public+base+过度生成+OCNLI过滤
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/new_ogfilter_output/组合好的用于NLI训练的traintestdev/"
--output_dir="./outputs/new_filter_legal_output/"

---------per_gpu_train_batch_size最初=16  epoch也需要调整  model_type跟着bert的类型调整  data_dir和output_dir也需要根据目标修改-------
--model_type=bert   
--model_name_or_path="/mnt/pretrained_language_models/bert-base-chinese/"
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/new_generate/组合好的用于NLI训练的traintestdev/"
--output_dir="./outputs/new_legal_output/"
--max_seq_length=128 
--per_gpu_train_batch_size=32 
--per_gpu_eval_batch_size=16 
--learning_rate=3e-5 
--num_train_epochs=20.0 
--task_name="ocnli"
--cache_dir="./cache_dir/"
--do_train 
--do_eval 
--do_lower_case 
--logging_steps=24487 
--save_steps=24487 
--overwrite_output_dir 
--seed=42
***************************************
****贪婪生成：
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/1648+400+1500+500/组合好的用于NLI训练的traintestdev/"
--output_dir="./outputs/legal_output/"
****过度生成&过滤：
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/ogfilter_output/组合完毕之后的用于NLI训练的traintestdev/"
--output_dir="./outputs/legal_output/"
****使用 微调了第11，pooler，classifier层的NLI分类器 过滤后的数据集
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/ogfilter_output_ft11pc/组合好的用于NLI训练的traintestdev/"
--output_dir="./outputs/legal_after_filter/"
****public+base
****public+base+贪婪生成
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/new_generate/组合好的用于NLI训练的traintestdev/"
--output_dir="./outputs/new_legal_output/"
****public+base+过度生成+OCNLI过滤
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/new_ogfilter_output/组合好的用于NLI训练的traintestdev/"
--output_dir="./outputs/new_filter_legal_output/"
************
-----------per_gpu_train_batch_size最初=16  epoch也需要调整  model_type跟着bert的类型调整  data_dir和output_dir也需要根据目标修改-------
python run_classifier.py \
--model_type=bert \
--model_name_or_path="/mnt/pretrained_language_models/bert-base-chinese/" \
--data_dir="/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/generate_data/new_ogfilter_output/组合好的用于NLI训练的traintestdev/" \
--output_dir="./outputs/new_filter_legal_output/" \
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
"""


"""----------------如果纯评估---------"""
"""--model_name_or_path是要使用的模型的路径---------
--model_name_or_path="./outputs/legal_output/nli_legal_best/"   这个就可以正常跑下来
--model_name_or_path="/mnt/pretrained_language_models/bert-base-chinese/"  这个就不行
----------------------------
--model_type=bert 
--model_name_or_path="/mnt/pretrained_language_models/bert-base-chinese/"
--data_dir="./CLUEdatasets/my_legal_nli/"
--output_dir="./outputs/test_output/"
--max_seq_length=128
--per_gpu_train_batch_size=16 
--per_gpu_eval_batch_size=32 
--learning_rate=3e-5 
--num_train_epochs=2.0 
--task_name="ocnli"
--do_eval 
--do_lower_case 
--logging_steps=24487 
--save_steps=24487 
--overwrite_output_dir 
--seed=42
-------------------------------------"""

"""--------------如果是要预测---------"""
"""----
--model_type=bert 
--model_name_or_path="./outputs/legal_output/nli_legal_best/" 
--data_dir="./CLUEdatasets/my_legal_nli/"
--output_dir="./outputs/test_output/"
--max_seq_length=128
--per_gpu_train_batch_size=16 
--per_gpu_eval_batch_size=16 
--learning_rate=3e-5 
--num_train_epochs=2.0 
--task_name="ocnli"
--do_predict 
--do_lower_case 
--logging_steps=24487 
--save_steps=24487 
--overwrite_output_dir 
--seed=42
*************************
python run_classifier.py \
--model_type=bert \
--model_name_or_path="/mnt/pretrained_language_models/bert-base-chinese/" \
--data_dir="./CLUEdatasets/my_legal_nli/" \
--output_dir="./outputs/ocnli_output/" \
--max_seq_length=128 \
--per_gpu_train_batch_size=16 \
--per_gpu_eval_batch_size=16 \
--learning_rate=3e-5 \
--num_train_epochs=2.0 \
--task_name="ocnli" \
--do_predict \
--do_lower_case \
--logging_steps=24487 \
--save_steps=24487 \
--overwrite_output_dir \
--seed=42
--------------------------------------"""
