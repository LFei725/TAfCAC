###
# target Data Format:
#{
#  'dockey': filename,
#  'doc_text': doc_text,
#  'sents':     [
#                  [chars in sent 0],
#                  [chars in sent 1],
#                   ...
#               ]
#  'total': money_str, e.g., "3900.0",
#  'vertexSet': [
#                 {'name': string, money_span_text,
#                  'sent_id': int, money in which sentence,
#                  'pos': [st, ed), postion of mention in a sentence,
#                  'label': 'Y/N'},
#                  {anthoer money entity}
#               ]
#}
###

import re
import os
import sys
import json
import numpy as np
from collections import Counter

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


def extractMoneys(doc_text):
    money_items = re.findall(r"\d*(?:\.\d+)?[一二三四五六七八九十百千万余]*元", doc_text)
    #if len(money_items) == 0:
    #    print(doc_text)
    #    print(money_items)

    full_money_items = []
    st_search = 0
    for money_item in money_items:
        if money_item == "元": continue
        st = doc_text.find(money_item, st_search)
        ed = st+len(money_item)
        st_search = ed
        #print(money_item, st, ed, doc_text[st:ed])
        assert money_item == doc_text[st:ed]
        full_money_items.append((money_item, st, ed, 'N'))
    return full_money_items


def extractMoneys2(doc_text):  # except "元"
    sentence_money_items = re.findall(r"\d*(?:\.\d+)?[一二三四五六七八九十百千万余]*元", doc_text)
    full_money_items = []
    st_search = 0
    for money_item in sentence_money_items:
        if money_item == "元":
            continue
        st = doc_text.find(money_item, st_search)
        ed = st + len(money_item) - 1
        st_search = ed
        assert money_item == doc_text[st:ed + 1]
        full_money_items.append((doc_text[st:ed], st, ed))
    return full_money_items


def create_dataset_files(input_dir, data_flag, debug=False):
    total_money = 0
    filename = input_dir + "/" + data_flag + ".json"

    char_set = set()
    doc_lens = []
    sent_lens = []

    doc_content = []
    for line_num, line in enumerate(open(filename, "r", encoding='utf-8').readlines()):
        input_json = json.loads(line)
        doc_key = input_json["filename"]
        doc_text = input_json["justice"]
        money = None
        if "money" in input_json:
            money = input_json["money"]
        ent_spans = input_json["zlabel"]

        sents = splitsentence(doc_text)
        new_sents = [] # segmented into chars, to be stored
        seqid2pos = dict() # used to convert ids
        ###### segment text into chars, keep moneys as one char [MONEY]
        sent_st = 0
        for sentid, sent in enumerate(sents):
            for char_seqid in range(len(sent)):
                seqid2pos[sent_st+char_seqid] = (sentid, char_seqid)
            # money_items = extractMoneys(sent)  # include "元"
            money_items = extractMoneys2(sent)   # except "元"
            if len(money_items) == 0:
                chars_sent = list(sent)
            else:  # replace money_text with [MONEY]
                chars_sent = []
                money_idx = dict([(money_item[1], money_item) for money_item in money_items])
                pointer = 0
                while(pointer<len(sent)):
                    if pointer not in money_idx:
                        seqid2pos[sent_st+pointer] = (sentid, len(chars_sent))
                        chars_sent.append(sent[pointer])
                        pointer += 1
                    else:
                        money_item = money_idx[pointer]
                        for i in range(pointer, money_item[2]):
                            seqid2pos[sent_st+i] = (sentid, len(chars_sent))
                        #chars_sent.append(money_item[0]) # money_text as a char
                        chars_sent.append("[money]")  # replace money_text with [MONEY]
                        pointer = money_item[2]
            new_sents.append(chars_sent)
            #print("#Sent{}:".format(sentid), chars_sent)
            sent_st += len(sent)

        for seqid, char in enumerate(doc_text):
            (new_sid, new_cid) = seqid2pos.get(seqid)
            if new_sents[new_sid][new_cid] != "[money]":
                assert new_sents[new_sid][new_cid].find(char) >= 0

        #chars = [char.lower() for char in doc_text]
        chars = [char.lower() for sent in new_sents for char in sent]  # all chars include [money]
        doc_lens.append(len(chars))
        for sent in new_sents:
            sent_lens.append(len(sent))

        vertexs = []
        for ent in ent_spans:
            st = ent[1]
            sent_id = seqid2pos[st][0]
            st = seqid2pos[st][1]
            ed = st + 1
            money_dict = {}
            money_dict["name"] = ent[0]
            money_dict["sent_id"] = sent_id
            money_dict["pos"] = [st, ed]
            money_dict["label"] = ent[-1]
            vertexs.append(money_dict)

        total_money += len(vertexs)
        char_set.update(set(chars))

        if debug:
            print("**********************Reading file", doc_key)
            print("#Text:", len(doc_text), doc_text)
            print("#Sent:", len(chars))
            for sentid, sent in enumerate(new_sents):
                print("#S{}:".format(sentid), sent)
        # create dict, add to output list
        output_json = {}
        output_json["dockey"] = doc_key
        output_json["doc_text"] = doc_text
        output_json["sents"] = new_sents
        output_json["total"] = money
        output_json["vertexSet"] = vertexs
        doc_content.append(output_json)
        #if line_num > 5: break

    # dumps output list
    if not os.path.exists(sys.argv[2]):
        os.mkdir(sys.argv[2])
    output_file = open(sys.argv[2] + "/" + data_flag + ".json", "w")
    output_file.write(json.dumps(doc_content, ensure_ascii=False))
    output_file.close()

    print('total number of money', total_money)
    print('Doc lens', max(doc_lens), Counter(doc_lens).most_common(5))
    print('Sent lens', max(sent_lens), Counter(sent_lens).most_common(5))
    return char_set

def create_vocabulary_files(char_set_train, char_set_test=None, char_set_dev=None):
    res_char_embeds = [line.rstrip().split() for line in open("data/char.embed", "r", encoding='utf-8').readlines()]
    res_char_embeds = dict([(item[0], item[1:]) for item in res_char_embeds])
    chars = (char_set_train.union(char_set_test).union(char_set_dev)).intersection(set(res_char_embeds.keys()))
    chars = sorted(list(chars))

    chars.insert(0, "BLANK")
    chars.insert(1, "UNK")
    chars.insert(2, "[money]")
    data_embeds = np.zeros((len(chars), 300))
    for i in range(3, len(chars)):
        embed = res_char_embeds.get(chars[i])
        embed = [float(item) for item in embed]
        data_embeds[i] = embed
    print("char embed size", data_embeds.shape)

    # matched word in 19kdocred_glove: train 85% dev 94% test 94%
    # full glove 400k: train 91% dev 96% test 96%
    #target_words = json.load(open("docred_word2id.json", "r")).keys()
    #target_words = [line.split()[0] for line in open("/home/qinyx/corpus/glove_embeddings/glove.6B.100d.txt", "r").readlines()]
    #matched_train = word_set_train.intersection(set(target_words))
    #print("matched train words in glove", len(matched_train), len(word_set_train), len(matched_train)*100.0/len(word_set_train), len(target_words))
    #matched_dev = word_set_dev.intersection(set(target_words))
    #print("matched dev words in glove", len(matched_dev), len(word_set_dev), len(matched_dev)*100.0/len(word_set_dev), len(target_words))
    #matched_test = set(word_set_test).intersection(set(target_words))
    #print("matched test words in glove", len(matched_test), len(word_set_test), len(matched_test)*100.0/len(word_set_test), len(target_words))
    #matched_test = set(word_set_test).intersection(set(matched_train))
    #print("matched test words in matched_train", len(matched_test), len(word_set_test), len(matched_test)*100.0/len(word_set_test), len(matched_train))

    char_id_dict = dict([(char, charid) for charid, char in enumerate(chars)])  # char:index
    rel_id_dict = {"N": 0, "A": 1, "D": 2, "S": 3, }

    vec_file = sys.argv[2]+"/vec"
    rel2id_file = open(sys.argv[2]+"/rel2id.json", "w")
    char2id_file = open(sys.argv[2]+"/token2id.json", "w")

    np.save(vec_file, data_embeds)
    rel2id_file.write(json.dumps(rel_id_dict))
    char2id_file.write(json.dumps(char_id_dict, ensure_ascii=False))
    rel2id_file.close()
    char2id_file.close()


if __name__=="__main__":
    input_dir = sys.argv[1]  # dir to raw money json files
    print("******************************* train file creating")
    char_set_train = create_dataset_files(input_dir, "train", debug=False)
    print("\n******************************* test file creating")
    char_set_test = create_dataset_files(input_dir, "test")
    print("\n******************************* dev file creating")
    char_set_dev = create_dataset_files(input_dir, "dev")

    create_vocabulary_files(char_set_train, char_set_test, char_set_dev)
