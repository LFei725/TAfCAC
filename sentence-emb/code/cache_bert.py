import torch
from transformers import *

import os
import time
import json
import h5py
import argparse

torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', type=str, default="prepro_data")
args = parser.parse_args()
out_path = args.out_path

pretrained_model_path="/mnt/pretrained_language_models/"
pretrained_nli_model_path="/mnt/3_new-try/task_augment/codes/now_use/my_classifier_pytorch/outputs/"


# 161行那里是控制是否可视化进度的print
class BertEncoder():
    def __init__(self):
        pretrained_weights_path = pretrained_model_path + 'bert-base-chinese'
        # pretrained_weights_path = pretrained_nli_model_path + 'new_filter_legal_output/nli_legal/'   # *=rams_bert_new_filter_nli_legal 已有
        # pretrained_weights_path = pretrained_nli_model_path + 'new_filter_legal_output/nli_legal+ocnli/'   # *_ocnli.h5 已有

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights_path)
        self.model = BertModel.from_pretrained(pretrained_weights_path).to(device)

    def encode_sent(self, words):
        # _convert_token_to_id
        input_ids = [self.tokenizer._convert_token_to_id(word) for word in words]
        print(input_ids)
        print(self.tokenizer.decode(input_ids))

        # tokenize
        tokenized_output = self.tokenizer(words, is_split_into_words=True)
        input_ids = tokenized_output["input_ids"]
        print(tokenized_output)
        print(input_ids)
        print(self.tokenizer.convert_ids_to_tokens(input_ids))
        print(self.tokenizer.decode(input_ids))

    # encode each token with convert_token_to_id
    def encode_sents_token(self, sentences, debug=False):  # 对于超过512的
        words = [word for sent in sentences for word in sent][:512]  # 只读取前512个字
        input_ids = [self.tokenizer._convert_token_to_id(word) for word in words]  # 这里没包括[CLS] [SEP] ，表示的是在字典里的index值
        oov_num = sum([1 if token_id == 100 else 0 for token_id in input_ids])
        # print(input_ids, self.tokenizer.decode(input_ids))
        input_ids = torch.tensor([input_ids]).to(device)
        with torch.no_grad():
            model_output = self.model(input_ids, output_hidden_states=True)
        bert_output = torch.mean(torch.stack(model_output.hidden_states[8:13]), 0)[0]
        # print(bert_output.shape) # (#word, 768)
        assert bert_output.shape[0] == len(words)
        return bert_output.cpu(), oov_num  # , len(words), oov_num

    # tokenize token into subtokens before encoding
    # input one sentence
    def encode_sent_token(self, sentence, debug=False):
        # print(sentences)
        words = [
            word.replace("👎🏻", '').replace("▪", '').replace('📸', '').replace("☰", '').replace("\x92", '').replace(
                "\x93", '').replace("\x94", '').replace("\x96", '').replace("\xad", '').replace("\x7f", '').replace(
                "\x9d", '').replace("\u200b", '').replace("\u200e", '').replace("�", '')
            for word in sentence]
        # for sent in sentences for word in sent]
        words = ["<UNK>" if len(w) == 0 else w for w in words]
        # print(words)
        if debug:
            print("Debug BERT encode_sents | words", len(words), words)
        tokenized_output = self.tokenizer(words, is_split_into_words=True)
        input_ids = tokenized_output["input_ids"]
        oov_num = sum([1 if token_id == 100 else 0 for token_id in input_ids])

        if debug:
            print("Debug BERT encode_sent_token | input_ids", input_ids)
        if (len(input_ids) > 512):
            return self.encode_sents_token(sentence)
        subtokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        subtokens = [w[2:] if w[:2] == "##" and len(w[2:]) > 0 else w for w in subtokens]
        assert len(subtokens) <= 512
        # print(tokenized_output)
        if debug:
            print("Debug BERT encode_sents | subtokens", subtokens)

        # obtain word_span: word:subtoken mapping
        word_start = [-1] * len(words)
        word_end = [-1] * len(words)
        j = 1
        word_start[0] = 1
        for i in range(len(words)):
            while (j < len(subtokens) - 1):
                if words[i].lower() == "".join(subtokens[word_start[i]:(j + 1)]) or "".join(
                        subtokens[word_start[i]:(j + 1)]) == '[UNK]':
                    word_end[i] = j
                    j += 1
                    if i < len(words) - 1:
                        word_start[i + 1] = j
                    break
                else:
                    j += 1
        if debug:
            print("Debug BERT encode_sents | word_start", word_start)
            print("Debug BERT encode_sents | word_end", word_end)
        assert word_end[-1] == len(subtokens) - 2
        assert -1 not in word_start
        assert -1 not in word_end
        word_span = [(word_start[i], word_end[i]) for i in range(len(words))]
        if debug:
            print("Debug BERT encode_sents | word_span", word_span)

        input_ids = torch.tensor([input_ids]).to(device)
        with torch.no_grad():
            model_output = self.model(input_ids, output_hidden_states=True)
        subtoken_reps = torch.mean(torch.stack(model_output.hidden_states[9:13]), 0).to(device)
        bert_output = []
        for i in range(len(words)):
            word_rep = torch.mean(subtoken_reps[0, word_span[i][0]:word_span[i][1] + 1, :], 0).to(device)
            bert_output.append(word_rep)
        bert_output = torch.vstack(bert_output).to(device)
        assert bert_output.shape[0] == len(words)
        return bert_output.cpu(), oov_num


def cache_rams_bert(bertEncoder, bert_output_file):
    train_oov=0
    test_oov=0
    with h5py.File(os.path.join(out_path, bert_output_file), "w") as bert_outfile:
        for suffix in ['train', 'test', 'dev']:
            word_num, oov_num = 0, 0
            ori_data = json.load(open(os.path.join(out_path, suffix + '.json'), "r", encoding='utf-8'))
            print("Cached bert representations for #doc", len(ori_data))
            for docid, inst in enumerate(ori_data):
                dockey = inst["dockey"]
                sents = inst["sents"]
                for sentid in range(len(sents)):
                    sent_rep, b = bertEncoder.encode_sent_token(sents[sentid])
                    word_num += len(sents[sentid])
                    oov_num += b
                    print(suffix, docid, dockey, sentid, sent_rep.shape)
                    try:
                        dataset = bert_outfile.create_dataset(dockey + str(sentid), data=sent_rep)
                        # print(dataset.shape)
                    except:
                        print("Skipping ", dockey, sentid)
                        continue
            print(suffix + "OOV ratio for token", oov_num, word_num, oov_num * 100.0 / word_num)
            if suffix == "train":
                train_oov = oov_num * 100.0 / word_num
            else:
                test_oov = oov_num * 100.0 / word_num
    print("train_oov=" + str(train_oov))
    print("test_oov=" + str(test_oov))
    print("Fished caching bert", time.asctime())


def load_h5(h5file, show_output=False):
    print("Start loading...", time.asctime())
    bert_outfile = h5py.File(os.path.join(out_path, h5file), "r")
    output = []
    for key in list(bert_outfile.keys()):
        output.append(bert_outfile[key][:])
        if show_output == True:
            print(key, '|', bert_outfile[key][:])
    print("Finished...", time.asctime())
    return output


if __name__ == "__main__":
    bertEncoder = BertEncoder()

    print("Start generate tokens...", time.asctime())
    bert_output_file = "rams_bert_base_chinese.h5"
    # bert_output_file = "rams_bert_new_filter_nli_legal.h5"
    # bert_output_file = "rams_bert_new_filter_nli_legal_ocnli.h5"
    cache_rams_bert(bertEncoder, bert_output_file)
    res = load_h5(bert_output_file)
    print("Found...", time.asctime())
