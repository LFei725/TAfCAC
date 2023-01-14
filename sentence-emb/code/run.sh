####  bash code/run.sh

## convert RAWRAW json file to money_items RAW json file
#python preprocess.py

## convert RAW json file to docred format json file
# cd home_directory
python raw_to_docred.py RawData ./data    

## prepare datacd/files for training models
cd code
cp ../data/token2id.json prepro_data
python gen_data.py --in_path ../data --out_path prepro_data 
cp ../data/*2id.json prepro_data
cp ../data/vec.npy prepro_data

##training:
# not adapting to use BERT
python train.py --save_name checkpoint_BiLSTM


python cache_bert.py
python train.py --save_name checkpoint_BiLSTM --use_bert


## only testing
#python train.py --eval --save_name checkpoint_BiLSTM --test_prefix test   # 先用这条命令
#python train.py --eval --save_name checkpoint_BERT_freeze --test_prefix test
