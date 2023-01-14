# coding: utf-8
import torch
import torch.nn as nn
from cn2an import cn2an
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import re
import os
import time
import datetime
import json
import sys
import h5py
import sklearn.metrics
from tqdm import tqdm
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import torch.nn.functional as F
from transformers import *
import cn2an

IGNORE_INDEX = -100
is_transformer = False


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


class raw_Config(object):
    def __init__(self, args):
        self.data_path = './prepro_data'
        self.use_gpu = True
        self.is_training = True
        self.max_length = 512
        self.max_sent_num = 50  # doc里含有的最多句子数
        self.max_vertex_num = 10  # doc里含有的最多money数
        self.relation_num = 66  # rams 66   好像是predict_re的第二维维度 [**,66]
        self.test_max_sent_num = 50

        self.coref_size = 20
        self.entity_type_size = 20
        self.max_epoch = 50
        self.opt_method = 'Adam'
        self.optimizer = None

        self.checkpoint_dir = './checkpoint'
        self.fig_result_dir = './fig_result'
        self.test_epoch = 1
        self.pretrain_model = None
        self.use_bert = False

        # graph based parameters
        self.use_graph = False
        self.max_node_num = 10
        self.graph_iter = 2
        self.graph_drop = 0.2
        self.max_edge_num = 5  # max edge type
        self.ablation = -1
        # self.max_neib_num = 12 # max neighbor number for each node
        # self.edge_pos = {0:0, 1:1, 2:2, 3:4, 4:5}
        self.max_neib_num = 20  # max neighbor number for each node
        self.edge_pos = {0: 0, 1: 1, 2: 2, 3: 4, 4: 9}

        self.epoch_range = None
        self.cnn_drop_prob = 0.5  # for cnn
        self.keep_prob = 0.8  # for lstm
        self.lstm_hidden_size = 128  # for lstm
        self.use_entity_type = False  # for lstm, rams
        self.use_coreference = False  # for lstm, rams
        self.lr = 0.001

        self.period = 50

        self.train_prefix = args.train_prefix
        self.test_prefix = args.test_prefix

        if not os.path.exists("log"):
            os.mkdir("log")

    def set_lr(self, lr):
        self.lr = lr

    def set_lstm_hidden_size(self, lstm_hidden_size):
        self.lstm_hidden_size = lstm_hidden_size

    def set_data_path(self, data_path):
        self.data_path = data_path

    def set_max_length(self, max_length):
        self.max_length = max_length

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes

    def set_window_size(self, window_size):
        self.window_size = window_size

    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_drop_prob(self, drop_prob):
        self.keep_prob = 1 - drop_prob

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def set_test_epoch(self, test_epoch):
        self.test_epoch = test_epoch

    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model

    def set_is_training(self, is_training):
        self.is_training = is_training

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_epoch_range(self, epoch_range):
        self.epoch_range = epoch_range

    def set_use_bert(self, use_bert):
        self.use_bert = use_bert

    def set_use_graph(self, use_graph):
        self.use_graph = use_graph

    def set_graph_drop(self, drop_prob):
        self.graph_drop = drop_prob

    def set_graph_iter(self, graph_iter):
        self.graph_iter = graph_iter

    def set_ablation(self, ablation):
        self.ablation = ablation

    def money_str2num(self, money_str):
        CN_NUM = {'一': 1.0, '二': 2.0, '三': 3.0, '四': 4.0, '五': 5.0, '六': 6.0, '七': 7.0, '八': 8.0, '九': 9.0}
        CN_UNIT = {'十': 10.0, '百': 100.0, '千': 1000.0, '万': 10000.0}
        money_str = money_str.replace('余', '')
        money_str = money_str.replace('元', '')

        length = len(money_str)  # length
        res = 0.0  # 结果

        if length == 0:
            return res

        # 仅剩一个字符，那直接把那个字符转换成小写，然后转为float即可
        if length == 1:
            if money_str[0] in CN_NUM:
                res = CN_NUM.get(money_str[0])
            elif money_str[0] in CN_UNIT:
                res = CN_UNIT.get(money_str[0])
            else:
                res = float(money_str[0])
            return res

        # 纯大写转小写or小写数字+大写单位。用了cn2an包。 ”7.1万元“也可以用。
        # https://www.jb51.net/article/206606.htm
        elif money_str[length - 1] > '9':
            try:
                res = cn2an.cn2an(money_str, "smart")
            except:
                res = 0.0
            res = float(res)
            return res

        # 纯小写
        res = float(money_str)
        return res

    def load_raw_data(self, file_name):
        raw_file = open(file_name, "r", encoding='utf-8')
        raw_data = []
        for line in raw_file.readlines():
            doc_item = json.loads(line)
            raw_data.append(doc_item)
        return raw_data

    def split_sentence(self, sentence):
        pattern = r'[，：；。]'  # []分割出来的列表不包括分隔符；()分割出来的列表包括分隔符
        s = sentence
        slist = re.split(pattern, s)
        return slist

    def rule_matching(self, doc_item):
        determine_list = ["认定取值为",
                          "共计价值人民币",
                          "共计价值为人民币",
                          "合计",
                          "价值共计人民币",
                          "共计人民币",
                          "合计价值人民币",
                          "共计价值",
                          "全部赃款",
                          "共计",
                          "价值合计人民币",
                          "价值共为人民币",
                          "价值人民币共计",
                          "合计价值",
                          "全部",
                          "实际得款",
                          "总计价值",
                          "总计",
                          "总价值",
                          "总共价值",
                          ]

        invalid_list = ["分得",
                        "扣押",
                        "赔偿",
                        "以人民币",
                        "元的价格",
                        "退赔",
                        "分给",
                        "缴获",
                        "元价格",
                        "罚金",
                        "退缴",
                        "退还",
                        "发还",
                        "退赃",
                        "退回",
                        "追回",
                        "查获",
                        "退出",
                        "赃款",
                        "归还",
                        "要求",
                        "补偿",
                        "查获",
                        "查扣",
                        "追还",
                        "追缴",
                        "虚构",
                        "供述",
                        "销赃",
                        "缴回",
                        "销赃",
                        "其中",
                        ]

        key_list = ["现金",
                    "价值",
                    "的人民币",
                    "骗取",  # 这个还需要好好斟酌一下
                    "骗得",
                    "转账",
                    "转款",
                    "给了",
                    "存了",
                    "取了",
                    "盗窃所得款",
                    "元现金",
                    "价格为",
                    "刷卡",
                    "人民币",
                    "金额",
                    "盗窃所得款",
                    "共计",
                    "合计",
                    "总计",
                    "订金",
                    "取值",
                    "取走",
                    "鉴定",
                    "诈骗",
                    "缴纳",
                    "骗走",
                    "刷取",
                    "支付",
                    "收取",
                    "取款",
                    "窃取",
                    "盗走",
                    ]

        money_items = doc_item["zlabel"]
        doc_text = doc_item["justice"]
        slist = self.split_sentence(doc_text)  # 获得最短句列表
        sl_index = 0
        sl_len = len(slist)
        determine_count = 0
        yes_count = 0

        for money_item in money_items:
            st = money_item[1]
            ed = money_item[2]
            scope_str = doc_text[st - 9:ed + 3]  # 判断有无关键词的范围区间
            for i in range(sl_index, sl_len):
                if money_item[0] in slist[i]:  # 在最短句里找到了金额
                    sl_index = i  # 更新开始索引值
                    len1 = len(money_item[0])
                    new_str = 'm' * len1
                    now_str = slist[i]
                    slist[sl_index] = now_str.replace(money_item[0], new_str, 1)
                    # 判断是否包含加法关键词
                    for j in range(len(key_list)):
                        if key_list[j] in scope_str:
                            money_item[3] = "Y"  # 找到了，则替换为Y
                            yes_count += 1
                            # 判断是否包含决定性词
                            for k in range(len(determine_list)):
                                if determine_list[k] in scope_str:
                                    determine_count += 1
                                    yes_count -= 1
                                    money_item[3] = "D"  # 找到了，则替换为D
                                    break  # 退出决定词表
                            # 判断是否是无效金额
                            for w in range(len(invalid_list)):
                                if invalid_list[w] in slist[i]:  # 如果找到了无效词
                                    if "全部赃款" not in scope_str:  # 如果没这个词，就确认无效
                                        if money_item[3] == "D":
                                            determine_count -= 1
                                            money_item[3] = "N"
                                        elif money_item[3] == "Y":
                                            yes_count -= 1
                                            money_item[3] = "N"
                                    break  # 退出无效词表
                            break  # 退出关键词表
                    break  # 退出最短句表

        # 如果只有一个决定性词，那就是那个值。。。。（也可能是总金额的一部分，先不考虑）
        if determine_count == 1:
            for money_item in money_items:
                if money_item[3] == "Y":
                    money_item[3] = "N"
                elif money_item[3] == "D":
                    money_item[3] = "Y"
        # 如果关键词=0，但决定词不为0，就把所有决定词换为关键词
        elif yes_count == 0 and determine_count != 0:
            for money_item in money_items:
                if money_item[3] == "D":
                    money_item[3] = "Y"
        # 否则
        else:
            for money_item in money_items:
                if money_item[3] == "D":
                    money_item[3] = "N"
        # 如果里面没找到关键词或者决定性词，就把所有的都换成Y，然后计算
        if determine_count + yes_count == 0:
            for money_item in money_items:
                money_item[3] = "Y"

        doc_item["zlabel"] = money_items
        return doc_item

    def get_rule_result(self, doc_item):
        doc_item = self.rule_matching(doc_item)  # 返回标注好的文档
        result = 0.0
        money_items = doc_item["zlabel"]
        for money_item in money_items:
            if money_item[3] == 'Y':
                money = self.money_str2num(money_item[0])  # 将string转为float
                result += money
        return int(result)

    def load_other_data(self):
        print("Reading token embedding and dictionaries...")
        # part of token embedding
        if not self.use_bert:  # 不用bert就是vec
            self.data_token_vec = np.load(os.path.join(self.data_path, 'vec.npy'))  # [word_count,300]
            # print(self.data_token_vec.shape)
        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.id2rel = {v: k for k, v in self.rel2id.items()}  # {0: 'N', 1: 'Y'}
        if self.use_graph:
            self.edge2id = json.load(open(os.path.join(self.data_path, 'edge2id.json')))
            self.id2edge = {v: k for k, v in self.edge2id.items()}

        # dictionary: dockey:bert_rep
        if self.use_bert:  # 用bert就是bertRep
            self.bertRep = h5py.File(os.path.join(self.data_path, "rams_bert4.h5"), "r")
        print("Finish reading")

    def load_train_data(self):
        print("Reading training data...")
        prefix = self.train_prefix

        self.data_train_token = np.load(os.path.join(self.data_path, prefix + '_token.npy'))  # prepro/train_token.npy shape=[doc_num, Max_sent_indoc, max_sent_length]
        self.train_file = json.load(open(os.path.join(self.data_path, prefix + '.json'),encoding='utf-8'))  # prepro/train.json
        if self.use_graph:
            self.data_train_neib = np.load(os.path.join(self.data_path, prefix + '_neib.npy'))
            self.data_train_edge = np.load(os.path.join(self.data_path, prefix + '_edge.npy'))
            # delete self-node edge
            self.data_train_neib[..., 0] = -1
            self.data_train_edge[..., 0] = -1
            if self.ablation == 1:  # adjacent sentence
                self.data_train_neib[..., 1] = -1
                self.data_train_edge[..., 1] = -1
            elif self.ablation == 2:  # adjacent tokens
                self.data_train_neib[..., 2:4] = -1
                self.data_train_edge[..., 2:4] = -1
            elif self.ablation == 3:  # coreference
                self.data_train_neib[..., 4:9] = -1
                self.data_train_edge[..., 4:9] = -1
            elif self.ablation == 4:  # syntactic deps
                self.data_train_neib[..., 9:] = -1  # - syntactic dep
                self.data_train_edge[..., 9:] = -1  # - syntactic dep
            elif self.ablation == 5:  # intra-sentence infor : adjW, syn
                self.data_train_neib[..., 2:4] = -1
                self.data_train_edge[..., 2:4] = -1
                self.data_train_neib[..., 9:] = -1  # - syntactic dep
                self.data_train_edge[..., 9:] = -1  # - syntactic dep
            elif self.ablation == 6:  # inter-sentence infor : adjS, coref
                self.data_train_neib[..., 1] = -1
                self.data_train_edge[..., 1] = -1
                self.data_train_neib[..., 4:9] = -1
                self.data_train_edge[..., 4:9] = -1

        print("Finish reading", prefix)

        self.train_len = ins_num = self.data_train_token.shape[0]  # 训练集里doc的个数
        assert (self.train_len == len(self.train_file))
        self.max_sent_num = self.data_train_token.shape[1]  # doc中最多的句子数

        self.train_order = list(range(ins_num))  # 就很单纯地搞了个序列,[0,1,2,...,train_doc_num]

    def load_test_data(self):
        print("Reading testing data...")

        prefix = self.test_prefix
        self.is_test = ('test' == prefix)
        self.data_test_token = np.load(os.path.join(self.data_path, prefix + '_token.npy'))  # [doc_num,Max_sent_indoc,max_sent_length]
        self.test_file = json.load(open(os.path.join(self.data_path, prefix + '.json'), encoding='utf-8'))  # test.json
        if self.use_graph:
            self.data_test_neib = np.load(os.path.join(self.data_path, prefix + '_neib.npy'))
            self.data_test_edge = np.load(os.path.join(self.data_path, prefix + '_edge.npy'))
            self.data_test_neib[..., 0] = -1
            self.data_test_edge[..., 0] = -1
            if self.ablation == 1:  # adjacent sentence
                self.data_test_neib[..., 1] = -1
                self.data_test_edge[..., 1] = -1
            elif self.ablation == 2:  # adjacent tokens
                self.data_test_neib[..., 2:4] = -1
                self.data_test_edge[..., 2:4] = -1
            elif self.ablation == 3:  # coreference
                self.data_test_neib[..., 4:9] = -1
                self.data_test_edge[..., 4:9] = -1
            elif self.ablation == 4:  # syntactic deps
                self.data_test_neib[..., 9:] = -1  # - syntactic dep
                self.data_test_edge[..., 9:] = -1  # - syntactic dep
            elif self.ablation == 5:  # intra-sentence infor : adjW, syn
                self.data_test_neib[..., 2:4] = -1
                self.data_test_edge[..., 2:4] = -1
                self.data_test_neib[..., 9:] = -1  # - syntactic dep
                self.data_test_edge[..., 9:] = -1  # - syntactic dep
            elif self.ablation == 6:  # inter-sentence infor : adjS, coref
                self.data_test_neib[..., 1] = -1
                self.data_test_edge[..., 1] = -1
                self.data_test_neib[..., 4:9] = -1
                self.data_test_edge[..., 4:9] = -1

        self.test_len = self.data_test_token.shape[0]  # doc_num
        assert (self.test_len == len(self.test_file))
        self.test_max_sent_num = self.data_test_token.shape[1]

        print("Finish reading", prefix)

    def get_train_batch(self):
        random.shuffle(self.train_order)  # 打乱顺序
        # word index in doc [max_sent_num,max_length]  先初始化一个最大的
        context_idxs = torch.LongTensor(self.max_sent_num, self.max_length).cuda()
        if self.use_graph:
            graph_neib = torch.LongTensor(self.max_node_num,
                                          self.max_neib_num).cuda()  # each node's neighbors(idx) in doc
            graph_edge = torch.LongTensor(self.max_node_num,
                                          self.max_neib_num).cuda()  # each node's edge(idx) to neighbors

        vertex_mapping = torch.Tensor(self.max_vertex_num, self.max_sent_num,
                                      self.max_length).cuda()  # each vertex span [max_vertex_num,max_sent_num,max_length] 也是先初始化一个最大的
        relation_mask = torch.Tensor(self.max_vertex_num).cuda()  # ent pair in doc [max_vertex_num]
        relation_label = torch.LongTensor(self.max_vertex_num).cuda()  # relation label idx

        for i, docid in enumerate(self.train_order):  # i: docid_in_iteration, docid: doc_index_in_rawData
            curr_graph_neib, curr_graph_edge = None, None
            for mapping in [vertex_mapping, relation_mask]:
                mapping.zero_()  # 置0处理
            relation_label.fill_(IGNORE_INDEX)   # 都填充上-100
            context_idxs.copy_(torch.from_numpy(self.data_train_token[docid, :, :]))  # 复制，这个时候还是[max_sent_count,all_max_len]
            if self.use_graph:
                graph_neib[i].copy_(torch.from_numpy(self.data_train_neib[docid, :, :]))
                graph_edge[i].copy_(torch.from_numpy(self.data_train_edge[docid, :, :]))

            ins = self.train_file[docid]  # prepro/train.json 中 第docid个doc的内容
            cur_s_len = len(ins['sents'])  # 当前doc的句子数量
            cur_v_num = len(ins['vertexSet'])  # 当前doc的money数

            for vid, vertex in enumerate(ins['vertexSet']):  # 遍历每一个money的vertexSet
                sid = vertex['sent_id']  # 所在句子id
                pos = vertex['pos']  # 所在的句子里面的起始index  eg:[8,9]
                label = vertex['label']  # 标签 1/0

                # average representation of tokens in span
                vertex_mapping[vid, sid, pos[0]:pos[1]] = 1.0 / (pos[1] - pos[0])  # 算是求个平均值吧
                # Plus representation of nearby tokens   #### 这里还可以改成加上旁边的一些字符求平均值
                # st = max(0, pos[0]-3)
                # ed = min(pos[1]+3, len(ins['sents'][sid]))
                # if ed == st: print(pos, len(ins['sents'][sid]))
                # vertex_mapping[vid, sid, st:ed] = 1.0 / (ed-st)
                relation_label[vid] = label
                relation_mask[vid] = 1

            if self.use_graph:
                curr_graph_neib = graph_neib[:cur_bsz, :, :].contiguous()
                curr_graph_edge = graph_edge[:cur_bsz, :, :].contiguous()

            input_lengths = (context_idxs[:cur_s_len] > 0).long().sum(dim=1)   # [当前doc的每个句子的长度]
            max_c_len = int(input_lengths.max())  # 当前doc的最长句子的长度

            yield {'context_idxs': context_idxs[:cur_s_len, :max_c_len].contiguous(),  # 缩减到[当前doc句子数，当前doc最长句子长度]
                   'graph_neib': curr_graph_neib,
                   'graph_edge': curr_graph_edge,
                   'vertex_mapping': vertex_mapping[:cur_v_num, :cur_s_len, :max_c_len],  # 缩减到[当前doc的money数，当前doc句子数，当前doc最长句子长度]
                   'relation_label': relation_label[:cur_v_num].contiguous(),
                   'input_lengths': input_lengths,
                   'relation_mask': relation_mask[:cur_v_num],
                   'dockey': ins['dockey'],  # dockeys in raw, used for bert
                   'index': docid,  # used for get bert representation, doc_ids in raw
                   # 'bert_subtokid': ins['bert_subtokid'],  # new add
                   # 'bert_tokenspan': ins['bert_tokenspan'],  # new add
                   }

    def get_test_batch(self):
        context_idxs = torch.LongTensor(self.test_max_sent_num, self.max_length).cuda()  # word index in doc
        if self.use_graph:
            graph_neib = torch.LongTensor(self.max_node_num,
                                          self.max_neib_num).cuda()  # each node's neighbors(idx) in doc
            graph_edge = torch.LongTensor(self.max_node_num,
                                          self.max_neib_num).cuda()  # each node's edge(idx) to neighbors

        vertex_mapping = torch.Tensor(self.max_vertex_num, self.test_max_sent_num,
                                      self.max_length).cuda()  # each vertex span
        relation_mask = torch.Tensor(self.max_vertex_num).cuda()  # ent pair in doc
        relation_label = torch.LongTensor(self.max_vertex_num).cuda()  # relation label idx

        for docid in range(self.test_len):  # docid: doc_id_inraw
            curr_graph_neib, curr_graph_edge = None, None
            for mapping in [vertex_mapping, relation_mask]:
                mapping.zero_()
            relation_label.fill_(IGNORE_INDEX)
            context_idxs.copy_(torch.from_numpy(self.data_test_token[docid, :, :]))
            if self.use_graph:
                graph_neib[i].copy_(torch.from_numpy(self.data_test_neib[docid, :, :]))
                graph_edge[i].copy_(torch.from_numpy(self.data_test_edge[docid, :, :]))

            ins = self.test_file[docid]
            cur_s_len = len(ins['sents'])
            cur_v_num = len(ins['vertexSet'])

            for vid, vertex in enumerate(ins['vertexSet']):
                sid = vertex['sent_id']
                pos = vertex['pos']
                label = vertex['label']
                vertex_mapping[vid, sid, pos[0]:pos[1]] = 1.0 / (pos[1] - pos[0])
                relation_label[vid] = label
                relation_mask[vid] = 1

            input_lengths = (context_idxs[:cur_s_len] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            if self.use_graph:
                curr_graph_neib = graph_neib[:cur_bsz, :, :].contiguous()
                curr_graph_edge = graph_edge[:cur_bsz, :, :].contiguous()

            yield {'context_idxs': context_idxs[:cur_s_len, :max_c_len].contiguous(),
                   'graph_neib': curr_graph_neib,
                   'graph_edge': curr_graph_edge,
                   'vertex_mapping': vertex_mapping[:cur_v_num, :cur_s_len, :max_c_len],  # [当前doc的money数，当前doc的句子数，当前doc的最长句子长度]
                   'relation_label': relation_label[:cur_v_num].contiguous(),
                   'input_lengths': input_lengths,
                   'relation_mask': relation_mask[:cur_v_num],
                   'dockey': ins['dockey'],  # dockeys in raw, used for bert
                   'index': docid,  # used for get bert representation, doc_ids in raw
                   # 'labels': labels, # only in test, label set in docs
                   # 'bert_subtokid': ins['bert_subtokid'],  # new add
                   # 'bert_tokenspan': ins['bert_tokenspan'],  # new add
                   }

    def train(self, model_pattern, model_name):  # model_pattern:BiLSTM model_name:checkpoint_BiLSTM
        print("Current training model will be saved to", model_name)
        params_arr = ["Hyper parameters: max_epoch:", self.max_epoch, "| optim:", self.opt_method, "lr:", self.lr,
                      "dropout:", 1 - self.keep_prob, " classify limit:", self.max_vertex_num,
                      "|| Model paramters: use_bert:", self.use_bert, "lstm_hd size:", self.lstm_hidden_size,
                      "| use_graph:", self.use_graph, "iter:", self.graph_iter, "graph_drop:", self.graph_drop,
                      "edge_num", self.max_edge_num, "ablate_edge:", self.ablation]
        logstr = " ------log:train=1648,test=400-----Y/N---"
        with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
            f_log.write("\n\n" + "------New train------" + time.asctime() + logstr + "\n")
            f_log.write(" ".join([str(item) for item in params_arr]) + "\n")
            print("------New train------" + time.asctime())
            print(logstr)
            print(params_arr)

        ori_model = model_pattern(config=self)   # 会进入BiLSTM，走一遍BiLSTM的__init__
        if self.pretrain_model != None:
            ori_model.load_state_dict(torch.load(self.pretrain_model))
        ori_model.cuda()
        model = ori_model

        # https://blog.csdn.net/fuyouzhiyi/article/details/89488232
        # 给定模型的参数model_parameters，利用filter排除所有model里面requires_grad=Fasle的参数，训练剩下的参数，这里的model是BiLSTM
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
        # optimizer = BertAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
        nll_average = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        # BCE = nn.BCEWithLogitsLoss(reduction='none')
        # BCE = nn.BCELoss(reduction='none')
        # softmax = nn.Softmax(dim=1)

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        best_f1 = 0.0
        best_epoch = 0

        model.train()  # 在使用pytorch构建神经网络的时候，训练过程中会在程序上方添加一句model.train()，作用是启用batch normalization和drop out

        global_step = 0
        total_loss = 0
        start_time = time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log/", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        # raw_train_data = self.load_raw_data("../RawData/train.json")

        for epoch in range(self.max_epoch):  # 对应每轮训练
            epoch_correct = 0
            epoch_predict = 0
            epoch_gold = 0
            batch_num = 0  # 记录doc的个数
            correct_money_num = 0
            # if epoch - best_epoch > 5:
            #    self.lr *= 0.8
            #    print("-- changed lr ", self.lr, " at epoch ", epoch)
            if epoch - best_epoch >= 10: break  # early stop
            for data in self.get_train_batch():  # 一个data是一个doc  乱序的，doc[index]对应raw_data里面的索引
                batch_num += 1

                context_idxs = data['context_idxs']  # [当前doc句子数，当前doc最长句子长度]
                graph_neib = data['graph_neib']  # None if not use_graph
                graph_edge = data['graph_edge']  # None if not use_graph
                vertex_mapping = data['vertex_mapping']  # [当前doc的money数，当前doc的句子数，当前doc的最长句子长度]
                relation_label = data['relation_label']  # [当前doc中每个money的label]
                input_lengths = data['input_lengths']  # 排好序了的各个句子的长度
                relation_mask = data['relation_mask']  # [1,1,...,1]
                dockey = data['dockey']  # 审判文书id
                index = data['index']  # doc_index in rawData

                # print("now is", dockey, batch_num)
                sents_count = len(input_lengths)  # 句子数
                # predict_re=[money_count，relation_num]   now relation_num=66
                predict_re = model(context_idxs, input_lengths, vertex_mapping, graph_neib, graph_edge, dockey, sents_count)  # 进入到module、BiLSTM。 predict_re=[money_count,relation_num]

                # CE loss
                loss = nll_average(predict_re.reshape(-1, self.relation_num), relation_label.reshape(-1))  # 交叉熵损失

                # pre_re_label = (predict_re>0).long()

                optimizer.zero_grad()  # 清空过往梯度
                # print("start loss.backward")
                # 这个地方会报错，linux直接报free()：invalid pointer Aborted(core dumped)，windows可能会停，也可能直接报错runtimeerror
                loss.backward()  # 反向传播，计算当前梯度  服务器内存要大一点才行，不然报错
                # loss.backward(retain_graph=True)  # 反向传播，计算当前梯度
                # print("loss.backward over")
                optimizer.step()  # 根据梯度更新网络参数
                # print("更新完毕")

                # torch.argmax(dim)会返回dim维度上张量最大值的索引。
                output = torch.argmax(predict_re, dim=-1)
                # 原本为1的个数。
                epoch_gold += torch.sum((relation_label > 0) * relation_mask).item()
                # 预测不为0的个数。torch.gt(a,b) 将a，b中元素相比较，a>b则返回1，否则返回0
                epoch_predict += torch.sum(torch.gt(output, 0) * relation_mask).item()  # 预测不为0的个数
                # epoch_predict += torch.sum(torch.eq(output, 1) * relation_mask).item()  # 预测为1的个数 结果和上面差不多
                # 预测的和原本的一致且为Y的个数      原本不为0 & 预测和原本相等   可以理解为成功预测出1的个数
                epoch_correct += torch.sum((torch.gt(relation_label, 0) & torch.eq(output, relation_label)) * relation_mask).item()

                # print("epooch***+++ooo")
                instance = self.train_file[index]  # 就是这个doc在prepro_data/train.json里面的样子
                total_money = 0.0
                for vid, vertex in enumerate(instance['vertexSet']):
                    pred_label = output[vid].cuda().item()
                    # print("Money {}: {}, label {}, predicted {}".format(vid, vertex['name'], vertex['label'], pred_label))
                    if pred_label == 1:  # 就是label==Y
                        money_value = vertex['name']
                        money_value = self.money_str2num(money_value)
                        total_money += money_value
                # print("Gold total_money {} with predicted {}".format(instance['total'], total_money))
                # add rules method
                # if total_money == 0:
                #     total_money = self.get_rule_result(raw_train_data[data['index']])

                if (str(total_money)) == instance['total']:
                    correct_money_num += 1

                global_step += 1
                total_loss += loss.item()
            # epoch_correct:成功预测出1的个数；epoch_predict:预测的不为0的个数；gold:原本为1的个数；
            epoch_pr = epoch_correct * 100.0 / (epoch_predict + (1e-10))  # precision，这里有点问题，不为0也不代表=1呀
            epoch_rec = epoch_correct * 100.0 / epoch_gold  # recall √
            epoch_f1 = 2 * epoch_pr * epoch_rec / (epoch_pr + epoch_rec + 1e-10)  # f1
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_epoch = epoch
                path = os.path.join(self.checkpoint_dir, model_name)
                torch.save(ori_model.state_dict(), path)
            logging(
                '| train epoch {:3d} | time: {:5.2f}s Loss {:5.5f} correct {:5.0f} predict {:5.0f} gold {:5.0f} Precision {:5.2f} Recall {:5.2f} F1 {:5.2f} Money_Precision {:5.2f}%'.format(
                    epoch, time.time() - start_time, total_loss / batch_num, epoch_correct, epoch_predict, epoch_gold,
                    epoch_pr, epoch_rec, epoch_f1, correct_money_num * 100.0 / self.train_len))
            start_time = time.time()
            total_loss = 0
            logging('| best epoch {:2d} with F1 {:5.3f}%'.format(best_epoch, best_f1))
            logging('-' * 89)

        print("Finish training" + time.asctime())
        print("Best epoch = %d | f1 = %f" % (best_epoch, best_f1))
        logging('Best epoch {:3d} | dev f1: {:.2f}'.format(best_epoch, best_f1))
        logging(time.asctime())
        print("Storing best result...")
        print("Finish storing")

    def test(self, model, model_name, output_file=False, output_analyze=False):
        eval_start_time = time.time()

        total_money_int_list = []
        total_loss = 0
        eval_metric = [0, 0, 0]  # correct, predict, gold

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        raw_test_data = self.load_raw_data("../RawData/test.json")

        prefix = "Test" if self.is_test else "Dev"

        nll_average = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        # BCE = nn.BCEWithLogitsLoss(reduction='none')
        # BCE = nn.BCELoss(reduction='none')
        # softmax = nn.Softmax(dim=1)
        batch_num = 0
        correct_money_num = 0
        rules_la_num = 0
        rules_correct = 0
        rules_compute = []
        rules_compute_success = []
        for data in self.get_test_batch():
            batch_num += 1
            with torch.no_grad():
                context_idxs = data['context_idxs']
                graph_neib = data['graph_neib']  # None if not use_graph
                graph_edge = data['graph_edge']  # None if not use_graph
                vertex_mapping = data['vertex_mapping']
                input_lengths = data['input_lengths']
                relation_mask = data['relation_mask']
                relation_label = data['relation_label']

                dockey = data['dockey']
                index = data['index']

                sents_count = len(input_lengths)
                predict_re = model(context_idxs, input_lengths, vertex_mapping, graph_neib, graph_edge, dockey, sents_count)
                # ce loss
                loss = nll_average(predict_re.reshape(-1, self.relation_num), relation_label.reshape(-1))
                total_loss += loss.item()

                # max(prob) -> label
                output = torch.argmax(predict_re, dim=-1)
                eval_metric[-1] += sum(relation_label.cpu()).item()  # gold
                eval_metric[1] += torch.sum(torch.gt(output, 0) * relation_mask).item()
                eval_metric[0] += torch.sum((torch.gt(relation_label, 0) & torch.eq(output, relation_label)) * relation_mask).item()

                if output_file:
                    instance = self.test_file[index]
                    single_doc_money = 0.0
                    sents = instance['sents']
                    if output_analyze:
                        print("*" * 50, index)
                        print("Doc: ", instance['doc_text'])
                    for vid, vertex in enumerate(instance['vertexSet']):
                        pred_label = output[vid].cuda().item()
                        if output_analyze:
                            print("Money {}: {}, label {}, predicted {}".format(vid, vertex['name'], vertex['label'], pred_label))  # eg:Money 0:510元，label 0，predicted 1
                        if pred_label == 1:
                            single_doc_money += self.money_str2num(vertex['name'])
                    # if single_doc_money == 0:
                    #     single_doc_money = self.get_rule_result(raw_test_data[data['index']])
                    if (str(single_doc_money)) == instance['total']:
                        correct_money_num += 1
                    # if single_doc_money == 0.0:
                    #     rules_la_num += 1
                    #     rules_compute.append(data['index'])
                    #     single_doc_money = self.get_rule_result(raw_test_data[data['index']])
                    #     if (str(single_doc_money)) == instance['total']:
                    #         rules_correct += 1
                    #         rules_compute_success.append(data['index'])
                    total_money_int_list.append(single_doc_money)  # int_money_list
                    # print("------------",str(single_doc_money),"---------------")

        # print("rules_la_num", rules_la_num)
        # print("rules_correct", rules_correct)
        # print("rules_compute", rules_compute)
        # print("rules_compute_success", rules_compute_success)

        if output_file:
            pred_file = open("log/preds_" + prefix + "_" + model_name, "w")
            raw_test_file = open("../RawData/raw_test.json", "r", encoding='utf-8')
            index = 0
            correct = 0
            for line in raw_test_file.readlines():  # 按行读取
                doc_item = json.loads(line)
                if str(total_money_int_list[index]) == doc_item['money']:
                    correct += 1
                index += 1
            print("Money_precision vs Raw_test_file:", correct * 100.0 / self.test_len)
            for item in total_money_int_list:
                int_item = int(item)
                # print(int_item)
                pred_file.write(json.dumps(int_item, ensure_ascii=False) + "\n")

        eval_metric.append(eval_metric[0] * 100.0 / (eval_metric[1] + 1e-10))
        eval_metric.append(eval_metric[0] * 100.0 / (eval_metric[2] + 1e-10))
        eval_metric.append(2 * eval_metric[-2] * eval_metric[-1] / (sum(eval_metric[-2:]) + 1e-10))
        logging(
            "| eval {:5} | Loss{:8.3f} correct {:5.0f} predict {:5.0f} gold {:5.0f} Precision {:5.2f} Recall {:5.2f} F1 {:5.2f}, Money_Precision {:5.2f}%".format(
                prefix, total_loss / batch_num, eval_metric[0], eval_metric[1], eval_metric[2], eval_metric[3],
                eval_metric[4], eval_metric[5], correct_money_num * 100.0 / self.test_len))
        return eval_metric

    def testall(self, model_pattern, model_name):
        print("Evaluating model from", model_name, time.asctime())
        params_arr = ["Hyper parameters: dropout:", 1 - self.keep_prob, " classify limit:", self.max_vertex_num,
                      "|| Model paramters: use_bert:", self.use_bert, "lstm_hd size:", self.lstm_hidden_size,
                      "| use_graph:", self.use_graph, "iter:", self.graph_iter, "graph_drop:", self.graph_drop,
                      "edge_num", self.max_edge_num, "ablate_edge:", self.ablation]
        print(params_arr)
        model = model_pattern(config=self)

        model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))
        model.cuda()
        # https://blog.csdn.net/weixin_43593330/article/details/107547202
        model.eval()  # 测试过程中会使用model.eval()，这时神经网络会沿用batch normalization的值，并不使用drop out
        self.test(model, model_name, True, False)
