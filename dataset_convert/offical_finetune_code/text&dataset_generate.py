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


torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:0"

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

MODEL_PATH_prefix = "./"

resentencesp = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')


# 将文档拆分成句子列表
def splitsentence(sentence):
    s = sentence
    slist = []
    for i in resentencesp.split(s):
        if resentencesp.match(i) and slist:
            slist[-1] += i
        elif i:
            slist.append(i)
    return slist


def data_convert(file_name):
    output_unlabeled_file = open(args.out_path + "unlabeled_" + file_name, "w", encoding='utf-8')
    output_all_NLI_file = open(args.out_path + "NLI_" + file_name, "w", encoding='utf-8')
    NLI_list = ["contradiction", "entailment", "neutral"]
    for nli_str in NLI_list:
        input_file = open(args.in_path + file_name, "r", encoding='utf-8')
        output_NLI_file = open(args.out_path + "public_NLI_" + nli_str + "_" + file_name, "w", encoding='utf-8')
        print("Reading from {}, writing to {} , {}"
              .format(input_file.name, output_NLI_file.name, output_unlabeled_file.name))
        print("Start generate tokens...", time.asctime())
        starttime = datetime.datetime.now()

        task_prefix = nli_str + ": "
        MODEL_PATH = MODEL_PATH_prefix + nli_str + "_public"
        tokenizer = T5PegasusTokenizer.from_pretrained(MODEL_PATH)
        model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

        index = 1
        pbar = progressbar.ProgressBar(n_total=len(input_file.readlines()), desc=file_name+":"+task_prefix)
        input_file = open(args.in_path + file_name, "r", encoding='utf-8')
        for line in input_file.readlines():
            doc_item = json.loads(line)
            doc_text = doc_item["justice"]
            sentences = splitsentence(doc_text)

            input_ids = tokenizer([task_prefix + sequence for sequence in sentences],
                                  padding=True,
                                  return_tensors="pt").input_ids.to(device)
            outputs = model.generate(input_ids,
                                     decoder_start_token_id=tokenizer.cls_token_id,
                                     eos_token_id=tokenizer.sep_token_id,
                                     max_length=300)
            result = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            generate_str = ""

            for i in range(len(sentences)):
                nli_item = {}
                nli_item["label"] = task_prefix[:-2]
                nli_item["sentence1"] = sentences[i]
                nli_item["sentence2"] = result[i].replace(" ", "")
                output_NLI_file.write(json.dumps(nli_item, ensure_ascii=False) + "\n")
                output_all_NLI_file.write(json.dumps(nli_item, ensure_ascii=False) + "\n")
                generate_str = generate_str + result[i].replace(" ", "")
            doc_item["justice"] = generate_str + "，"
            output_unlabeled_file.write(json.dumps(doc_item, ensure_ascii=False) + "\n")
            pbar(index-1)
            index += 1

        print("writing to {} over".format(output_NLI_file.name))
        print("finish time...", time.asctime())
        endtime = datetime.datetime.now()
        print("writing to" + output_NLI_file.name + "   use: " + str((endtime - starttime).seconds) + "s")

    print("writing to {} and {} over".format(output_unlabeled_file.name, output_all_NLI_file.name))
    print("finish time...", time.asctime())


if __name__ == "__main__":
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    data_convert("train.json")
    data_convert("test.json")
    data_convert("fu_train.json")
    data_convert("chu_test.json")

