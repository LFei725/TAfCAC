import datetime
import re
import os
import json
import argparse
import time

import torch

from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import BertTokenizer
import jieba
from tools import progressbar
import tqdm

torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:0"
"""train:contradiction: 3:00:00"""  # 3*9=21


# 1+2+3*(3*2)=21
class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default='../raw_data/', help='input directory')
parser.add_argument('--out_path', type=str, default='../generate_data/', help='output directory')
args = parser.parse_args()

MODEL_PATH_prefix = "/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/offical_finetune_code/"

resentencesp = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')


def splitsentence(sentence):
    s = sentence
    slist = []
    for i in resentencesp.split(s):
        if resentencesp.match(i) and slist:
            slist[-1] += i
        elif i:
            slist.append(i)
    return slist


def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open("log/generate_21_batch", 'a+') as f_log:
            f_log.write(s + '\n')


def data_convert(file_name):
    output_all_NLI_file = open(args.out_path + "NLI_" + file_name, "w", encoding='utf-8')
    NLI_list = ["contradiction", "entailment", "neutral"]
    for nli_str in NLI_list:
        input_file = open(args.in_path + file_name, "r", encoding='utf-8')
        output_NLI_file = open(args.out_path + "NLI_" + nli_str + "_" + file_name, "w",
                               encoding='utf-8')
        print("Reading from {}, writing to {} ".format(input_file.name, output_NLI_file.name))
        print("Start generate tokens...", time.asctime())
        starttime = datetime.datetime.now()

        task_prefix = nli_str + ": "
        MODEL_PATH = MODEL_PATH_prefix + nli_str + "_public"
        tokenizer = T5PegasusTokenizer.from_pretrained(MODEL_PATH)
        model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

        index = 1
        start_i = 1
        pbar = progressbar.ProgressBar(n_total=len(input_file.readlines()), desc=file_name + ":" + task_prefix)
        input_file = open(args.in_path + file_name, "r", encoding='utf-8')
        for line in input_file.readlines():
            if index == 540:
                index += 1
                output_NLI_file.write("--------\n")
                continue

            doc_item = json.loads(line)
            doc_text = doc_item["justice"]
            sentences = splitsentence(doc_text)
            all_sentence_num = 21
            sentences_result = [["" for x in range(all_sentence_num)] for y in range(len(sentences))]

            input_ids = tokenizer([task_prefix + sequence for sequence in sentences],
                                  padding=True,
                                  return_tensors="pt").input_ids.to(device)
            greedy_output = model.generate(input_ids,
                                           decoder_start_token_id=tokenizer.cls_token_id,
                                           eos_token_id=tokenizer.sep_token_id,
                                           max_length=256)
            now_result = tokenizer.batch_decode(greedy_output, skip_special_tokens=True)
            i = 0
            for j in range(len(now_result)):
                sentences_result[i][0] = now_result[j].replace(" ", "")
                i += 1

            beam_output10 = model.generate(input_ids,
                                           decoder_start_token_id=tokenizer.cls_token_id,
                                           eos_token_id=tokenizer.sep_token_id,
                                           num_beams=10,
                                           num_return_sequences=2,
                                           no_repeat_ngram_size=2,
                                           early_stopping=True,
                                           max_length=256)
            for i, beam_output in enumerate(beam_output10):
                sentences_result[i // 2][1 + i % 2] = ''.join(
                    tokenizer.decode(beam_output, skip_special_tokens=True)).replace(' ', '')

            sentence_index_index = 3
            seed_list = [-156453, 85324, 3652]  # 1+2+3*(3*2)=21
            for seed in seed_list:
                torch.random.manual_seed(seed)
                per_gen_num = 2
                sample_output_t = model.generate(input_ids,
                                                 decoder_start_token_id=tokenizer.cls_token_id,
                                                 eos_token_id=tokenizer.sep_token_id,
                                                 do_sample=True,
                                                 num_return_sequences=per_gen_num,
                                                 max_length=256,
                                                 temperature=0.7,
                                                 top_k=0)
                for i, sample_output_t1 in enumerate(sample_output_t):
                    sentences_result[i // per_gen_num][sentence_index_index + i % per_gen_num] = ''.join(
                        tokenizer.decode(sample_output_t1, skip_special_tokens=True)).replace(' ', '')
                sentence_index_index += per_gen_num

                sample_output_top_k = model.generate(input_ids,
                                                     decoder_start_token_id=tokenizer.cls_token_id,
                                                     eos_token_id=tokenizer.sep_token_id,
                                                     do_sample=True,
                                                     num_return_sequences=per_gen_num,
                                                     max_length=256,
                                                     top_k=50)
                for i, sample_output_top_k1 in enumerate(sample_output_top_k):
                    sentences_result[i // per_gen_num][sentence_index_index + i % per_gen_num] = ''.join(
                        tokenizer.decode(sample_output_top_k1, skip_special_tokens=True)).replace(' ', '')
                sentence_index_index += per_gen_num

                sample_output_top_p = model.generate(input_ids,
                                                     decoder_start_token_id=tokenizer.cls_token_id,
                                                     eos_token_id=tokenizer.sep_token_id,
                                                     do_sample=True,
                                                     num_return_sequences=per_gen_num,
                                                     max_length=256,
                                                     top_p=0.92,
                                                     top_k=0)
                for i, sample_output_top_p1 in enumerate(sample_output_top_p):
                    sentences_result[i // per_gen_num][sentence_index_index + i % per_gen_num] = ''.join(
                        tokenizer.decode(sample_output_top_p1, skip_special_tokens=True)).replace(' ', '')
                sentence_index_index += per_gen_num
            assert sentence_index_index == all_sentence_num

            for i in range(len(sentences) * all_sentence_num):
                nli_item = {}
                nli_item["label"] = task_prefix[:-2]
                nli_item["sentence1"] = sentences[i // all_sentence_num]
                nli_item["sentence2"] = sentences_result[i // all_sentence_num][i % all_sentence_num]
                output_NLI_file.write(json.dumps(nli_item, ensure_ascii=False) + "\n")
                output_all_NLI_file.write(json.dumps(nli_item, ensure_ascii=False) + "\n")

            pbar(index - 1)
            index += 1
            torch.cuda.empty_cache()

        print("\nwriting to {} over".format(output_NLI_file.name))
        print("finish time...", time.asctime())
        endtime = datetime.datetime.now()
        print("writing to" + output_NLI_file.name + "---use: " + str((endtime - starttime).seconds) + "s")
        logging("use model: " + MODEL_PATH)
        logging("generate file: " + output_NLI_file.name)
        logging("from index" + str(start_i) + " to end" + " cost: " + str((endtime - starttime).seconds) + "s")

    print("finish time...", time.asctime())


if __name__ == "__main__":
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    logging("\n\n------------------New Generation----------------" + time.asctime() + "\n")
    # data_convert("train.json")
    # data_convert("test.json")
    data_convert("fu_train.json")
    # data_convert("chu_test.json")
