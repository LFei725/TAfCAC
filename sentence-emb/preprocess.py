import re
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default='./RawData/', help='input directory')
parser.add_argument('--out_path', type=str, default='./RawData/', help='output directory')
args = parser.parse_args()

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


def preprocess_doc(json_str):
    doc_item = json.loads(json_str)
    doc_text = doc_item["justice"]
    doc_text = re.sub("[\n ]+", "", doc_text)
    sentences = splitsentence(doc_text)

    doc_item["justice"] = doc_text
    return doc_item


def extractMoneys(doc_item):
    doc_text = doc_item["justice"]
    money_items = re.findall(r"\d*(?:\.\d+)?[一二三四五六七八九十百千万余]*元", doc_text)
    if len(money_items) == 0:
        print(doc_text)
        print(money_items)

    full_money_items = []
    st_search = 0
    for money_item in money_items:
        if money_item == "元": continue
        st = doc_text.find(money_item, st_search)
        ed = st + len(money_item)
        st_search = ed
        assert money_item == doc_text[st:ed]
        full_money_items.append((money_item, st, ed, 'N'))
    return full_money_items


def preprocess(str1):
    input_file = open(args.in_path + "raw_" + str1 + ".json", "r", encoding='utf-8')
    output_file = open(args.out_path + str1 + ".json", "w", encoding='utf-8')
    print("Reading from {}, writing to {}".format(input_file.name, output_file.name))

    for line in input_file.readlines():
        doc_item = preprocess_doc(line.strip())
        money_items = extractMoneys(doc_item)
        doc_item["zlabel"] = money_items
        output_file.write(json.dumps(doc_item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    # print("*****preprocess raw train data*****")
    # preprocess("train")
    print("*****preprocess raw test data*****")
    preprocess("test")
    print("*****complete preprocess raw data*****")
