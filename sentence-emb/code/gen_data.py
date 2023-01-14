import numpy as np
import os
import json
import argparse
from transformers import *

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default="../data")
parser.add_argument('--out_path', type=str, default="prepro_data")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path

train_annotated_file_name = os.path.join(in_path, 'train.json')
test_file_name = os.path.join(in_path, 'test.json')
dev_file_name = os.path.join(in_path, 'dev.json')
rel2id = json.load(open(os.path.join(in_path, 'rel2id.json'), "r"))


BERT_PATH = '/mnt/pretrained_language_models/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)


# max_length is max number of tokens in a sent
def init(data_file_name, rel2id, max_length=512, suffix=''):
    total_arg_num = 0
    Max_vertex = 0
    Max_sent_indoc = 0
    Max_token_insent = 0
    data = []  # store all data here, str to index
    token_dim = 300
    if_bert_tokenize_doc = 'false'

    ori_data = json.load(open(data_file_name))

    # store updated data as list of dict items (dockey, sents, vertexSet: update pos_in_sent to pos_in_doc, lables: update relation to relid, add Ls: sent start index, add na_tripple: negative triples, add trigger: span)
    for i in range(len(ori_data)):
        # calculate start index of each sentences
        Ls = [0]
        L = 0
        sorted_sents = sorted([(sid, sent) for sid, sent in enumerate(ori_data[i]['sents'])], key=lambda x: len(x[1]), reverse=True)
        sentid_ori2curr = dict([(sorted_sents[new_id][0], new_id) for new_id in range(len(sorted_sents))])
        sents = []
        for sid, sent in sorted_sents:
            sent_len = min(max_length, len(sent))
            L += sent_len
            Ls.append(L)
            Max_token_insent = max(Max_token_insent, sent_len)
            sents.append(sent[:sent_len])
        ori_data[i]['sents'] = sents

        # point position added with sent start position
        # label convert to labelid
        vertexSet = ori_data[i]['vertexSet']
        for j in range(len(vertexSet)):
            pre_sent_id = vertexSet[j]['sent_id']
            sent_id = sentid_ori2curr[pre_sent_id]
            (pos1, pos2) = vertexSet[j]['pos']
            if pos2 >= max_length:
                print("[Warning!!] Deleted vertex because position exceed max_length", vertexSet[j])
                continue
            label = vertexSet[j]['label']
            vertexSet[j]['sent_id'] = sent_id
            vertexSet[j]['label'] = rel2id[label]
        ori_data[i]['vertexSet'] = vertexSet
        if_bert_tokenize_doc = 'true'

        item = {}  # used to store info in current document
        item['dockey'] = ori_data[i]['dockey']
        item['doc_text'] = ori_data[i]['doc_text']
        item['sents'] = sents  # truncated to max_length
        item['total'] = ori_data[i]['total']
        item['vertexSet'] = vertexSet  # modified
        item['Ls'] = Ls
        data.append(item)

        total_arg_num += len(item['vertexSet'])
        Max_vertex = max(Max_vertex, len(vertexSet))
        Max_sent_indoc = max(Max_sent_indoc, len(ori_data[i]['sents']))

    print('#Doc num:', len(ori_data))
    print('Max_Sent_Indoc:', Max_sent_indoc)
    print('Max_Token_Insent:', Max_token_insent)
    print('Max_Vertex:', Max_vertex)
    print('Total vertex num', total_arg_num)

    # saving train/test.json
    print("Saving file", suffix, ".json")
    json.dump(data, open(os.path.join(out_path, suffix + '.json'), "w", encoding='utf-8'), ensure_ascii=False)

    ############################################################
    ## convert token to id according to char2id.json, store in _token.npy
    token2id = json.load(open(os.path.join(out_path, "token2id.json")))

    # convert from str to id
    # ins_token: store doc_token [doc_num, max_sent_num, max_length] padding with BLANK
    doc_num = len(data)
    ins_token = np.zeros((doc_num, Max_sent_indoc, max_length), dtype=np.int64)  #### [doc_num,max_sent_count,max_sent_len]

    for i in range(len(data)):
        item = data[i]
        for sid, sent in enumerate(item['sents']):
            for tokid, token in enumerate(sent):
                token = token.lower()
                if token in token2id:
                    ins_token[i][sid][tokid] = token2id[token]
                else:
                    ins_token[i][sid][tokid] = token2id['UNK']

    print("---Finishing processing---")
    print("token_dim =", token_dim)
    print("use_bert_tokenize_doc =", if_bert_tokenize_doc)
    np.save(os.path.join(out_path, suffix + '_token.npy'), ins_token)
    print("Finish saving", suffix, "_token.npy")


def bert_tokenize_doc(sentences, debug=False):
    words = [word.replace("👎🏻", '').replace("▪", '').replace('📸', '').replace("☰", '').replace(
        "\x92", '').replace("\x93", '').replace("\x94", '').replace("\x96", '').replace("\xad", '').replace(
        "\x7f", '').replace("\x9d", '').replace("\u200b", '').replace("\u200e", '').replace("�", '')
             for sent in sentences for word in sent]
    words = ["<UNK>" if len(w) == 0 else w for w in words]
    tokenized_output = tokenizer(words, is_split_into_words=True)
    input_ids = tokenized_output["input_ids"]
    if (len(input_ids) > 512):
        input_ids = [tokenizer._convert_token_to_id(word) for word in words[:510]]
        input_ids.insert(0, 101)  # [CLS]
        input_ids.append(102)  # [SEP]
        word_span = [(i + 1, i + 1) for i in range(len(words))]
        word_span[0] = (0, 1)
        word_span[-1] = (len(words), len(words) + 1)
    else:
        subtokens = tokenizer.convert_ids_to_tokens(input_ids)  # word + [ money ]
        subtokens = [w[2:] if w[:2] == "##" and len(w[2:]) > 0 else w for w in subtokens]
        assert len(subtokens) <= 512

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
        assert word_end[-1] == len(subtokens) - 2
        assert -1 not in word_start
        assert -1 not in word_end
        word_start[0] = 0  # include [cls]
        word_end[-1] = len(subtokens) - 1  # include [sep]
        word_span = [(word_start[i], word_end[i]) for i in range(len(words))]
    input_ids.extend([0] * (512 - len(input_ids)))
    return input_ids, word_span


init(train_annotated_file_name, rel2id, max_length=512, suffix='train')
init(test_file_name, rel2id, max_length=512, suffix='test')
init(dev_file_name, rel2id, max_length=512, suffix='dev')
