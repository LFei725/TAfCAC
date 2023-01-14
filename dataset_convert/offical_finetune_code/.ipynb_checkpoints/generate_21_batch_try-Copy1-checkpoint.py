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
"""-------------------------------------需要 24G 的  服务器---------------------------------------------"""
"""-------------------------------------------------需要检查out_path的生成位置---------------------------"""


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
parser.add_argument('--out_path', type=str, default='../generate_data/new_overgenerate/', help='output directory')
args = parser.parse_args()

MODEL_PATH_prefix = "/mnt/3_new-try/task_augment/codes/now_use/text2text-generate/offical_finetune_code/"

resentencesp = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')


# 将文档拆分成句子列表
def splitsentence(sentence):
    s = sentence
    slist = []
    for i in resentencesp.split(s):  # 以正则表达式中的符号为分隔符
        if resentencesp.match(i) and slist:  # 如果当前字符串为正则表达式中的符号，且slist非空
            slist[-1] += i  # 就在slist中的最后一个字符串的最后拼接上当前字符串。
        elif i:
            slist.append(i)  # 否则就把字符串添加到列表中
    return slist


# 接下来读数据，读数据集中拆分好的每个句子吧，然后丢进去转换，暂时只能一个一个丢进去，然后只生成3个过度生成的文件
def data_convert(file_name):
    NLI_list = ["contradiction", "entailment", "neutral"]
    for nli_str in NLI_list:
        input_file = open(args.in_path + file_name, "r", encoding='utf-8')
        output_NLI_file = open(args.out_path + "NLI_394041" + nli_str + "_" + file_name, "w", encoding='utf-8')  # eg. NLI_contradiction_train.json
        print("Reading from {}, writing to {} ".format(input_file.name, output_NLI_file.name))
        print("Start generate tokens...", time.asctime())
        starttime = datetime.datetime.now()

        task_prefix = nli_str + ": "
        # MODEL_PATH = MODEL_PATH_prefix + nli_str + "_res_pe"
        MODEL_PATH = MODEL_PATH_prefix + nli_str + "_public"
        tokenizer = T5PegasusTokenizer.from_pretrained(MODEL_PATH)
        model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

        index = 1
        pbar = progressbar.ProgressBar(n_total=len(input_file.readlines()), desc=file_name + ":" + task_prefix)
        input_file = open(args.in_path + file_name, "r", encoding='utf-8')
        for line in input_file.readlines():  # 读doc
            # print("start process " + file_name + "---" + str(index))
            if index < 539:  # 只生成 539 540 541
                index += 1
                continue
            if index > 541:
                break
            doc_item = json.loads(line)
            doc_text = doc_item["justice"]
            sentences = splitsentence(doc_text)
            all_sentence_num = 21
            sentences_result = [["" for x in range(all_sentence_num)] for y in range(len(sentences))]
            sentence_index = 0
            sentence_index_index = 0

            input_ids = tokenizer([task_prefix + sequence for sequence in sentences],
                                  padding=True,
                                  return_tensors="pt").input_ids.to(device)
            greedy_output = model.generate(input_ids,  # model_output_id 就直接是output的token_id了，然后经过分词器解码就可以输出了
                                           decoder_start_token_id=tokenizer.cls_token_id,  # 标记解码器输出的起始位置
                                           eos_token_id=tokenizer.sep_token_id,
                                           max_length=256)  # 最大输出长度
            now_result = tokenizer.batch_decode(greedy_output, skip_special_tokens=True)
            i = 0
            for j in range(len(now_result)):
                sentences_result[i][0] = now_result[j].replace(" ", "")
                i += 1

            beam_output10 = model.generate(input_ids,
                                           decoder_start_token_id=tokenizer.cls_token_id,
                                           eos_token_id=tokenizer.sep_token_id,
                                           num_beams=10,  # 保持10个束
                                           num_return_sequences=2,  # 返回3个最可能的序列
                                           no_repeat_ngram_size=2,  # 不允许2词单词重复出现第二次
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
                                                 temperature=0.7,  # 值越小越趋向于贪婪解码
                                                 top_k=0)  # 激活sampling，通过设置 top_k=0 停用 Top-K 采样
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
            assert sentence_index_index == all_sentence_num  # 确定每个句子生成了50个

            for i in range(len(sentences) * all_sentence_num):
                nli_item = {}
                nli_item["label"] = task_prefix[:-2]
                nli_item["sentence1"] = sentences[i // all_sentence_num]
                nli_item["sentence2"] = sentences_result[i // all_sentence_num][i % all_sentence_num]
                output_NLI_file.write(json.dumps(nli_item, ensure_ascii=False) + "\n")

            pbar(index - 1)
            index += 1
            torch.cuda.empty_cache()

        print("\nwriting to {} over".format(output_NLI_file.name))
        print("finish time...", time.asctime())
        endtime = datetime.datetime.now()
        print("writing to" + output_NLI_file.name + "---use: " + str((endtime - starttime).seconds) + "s")

    print("finish time...", time.asctime())


if __name__ == "__main__":
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

#     data_convert("train.json")
#     data_convert("test.json")
    data_convert("fu_train.json")
#     data_convert("chu_test.json")
