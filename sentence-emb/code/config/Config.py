# coding: utf-8
import csv
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
import copy

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import torch.nn.functional as F
from transformers import *
import cn2an
import wandb

IGNORE_INDEX = -100
is_transformer = False


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


class Config(object):
    def __init__(self, args):
        self.data_path = './prepro_data'
        self.use_gpu = True
        self.is_training = True
        self.max_length = 512
        self.max_sent_num = 50
        self.max_vertex_num = 10
        self.relation_num = 4
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
        self.dev_prefix = args.dev_prefix

        self.acc_total = Accuracy()

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

        length = len(money_str)
        res = 0.0

        if length == 0:
            return res

        if length == 1:
            if money_str[0] in CN_NUM:
                res = CN_NUM.get(money_str[0])
            elif money_str[0] in CN_UNIT:
                res = CN_UNIT.get(money_str[0])
            else:
                res = float(money_str[0])
            return res

        elif money_str[length - 1] > '9':
            try:
                res = cn2an.cn2an(money_str, "smart")
            except:
                res = 0.0
            res = float(res)
            return res

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
        pattern = r'[，：；。]'
        s = sentence
        slist = re.split(pattern, s)
        return slist

    def load_other_data(self):
        print("Reading token embedding and dictionaries...")
        # part of token embedding
        if not self.use_bert:
            self.data_token_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
            # print(self.data_token_vec.shape)
        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        if self.use_graph:
            self.edge2id = json.load(open(os.path.join(self.data_path, 'edge2id.json')))
            self.id2edge = {v: k for k, v in self.edge2id.items()}

        # dictionary: dockey:bert_rep
        if self.use_bert:
            self.bertRep = h5py.File(os.path.join(self.data_path, "rams_bert_base_chinese.h5"), "r")
#             self.bertRep = h5py.File(os.path.join(self.data_path, "rams_bert_new_filter_nli_legal.h5"), "r")
#             self.bertRep = h5py.File(os.path.join(self.data_path, "rams_bert_new_filter_nli_legal_ocnli.h5"), "r")
            print("当前使用的bert h5文件:--" + self.bertRep.filename)
        print("Finish reading")

    def load_train_data(self):
        print("Reading training data...")
        prefix = self.train_prefix

        self.data_train_token = np.load(os.path.join(self.data_path, prefix + '_token.npy'))
        self.train_file = json.load(open(os.path.join(self.data_path, prefix + '.json'), encoding='utf-8'))
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
        self.train_len = ins_num = self.data_train_token.shape[0]
        assert (self.train_len == len(self.train_file))
        self.max_sent_num = self.data_train_token.shape[1]
        self.train_order = list(range(ins_num))

    def load_test_data(self):
        print("Reading testing data...")

        prefix = self.test_prefix
        self.is_test = ('test' == prefix)
        self.data_test_token = np.load(
            os.path.join(self.data_path, prefix + '_token.npy'))
        self.test_file = json.load(open(os.path.join(self.data_path, prefix + '.json'), encoding='utf-8'))
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

    def load_dev_data(self):
        print("Reading dev data...")
        prefix = self.dev_prefix
        self.data_dev_token = np.load(
            os.path.join(self.data_path, prefix + '_token.npy'))
        self.dev_file = json.load(
            open(os.path.join(self.data_path, prefix + '.json'), encoding='utf-8'))
        self.dev_len = ins_num = self.data_dev_token.shape[0]
        assert (self.dev_len == len(self.dev_file))
        self.dev_max_sent_num = self.data_dev_token.shape[1]
        self.dev_order = list(range(ins_num))
        print("Finish reading", prefix)

    def get_train_batch(self):
        random.shuffle(self.train_order)
        context_idxs = torch.LongTensor(self.max_sent_num, self.max_length).cuda()
        if self.use_graph:
            graph_neib = torch.LongTensor(self.max_node_num,
                                          self.max_neib_num).cuda()  # each node's neighbors(idx) in doc
            graph_edge = torch.LongTensor(self.max_node_num,
                                          self.max_neib_num).cuda()  # each node's edge(idx) to neighbors

        vertex_mapping = torch.Tensor(self.max_vertex_num, self.max_sent_num,
                                      self.max_length).cuda()
        relation_mask = torch.Tensor(self.max_vertex_num).cuda()  # ent pair in doc [max_vertex_num]
        relation_label = torch.LongTensor(self.max_vertex_num).cuda()  # relation label idx

        for i, docid in enumerate(self.train_order):  # i: docid_in_iteration, docid: doc_index_in_rawData
            curr_graph_neib, curr_graph_edge = None, None
            for mapping in [vertex_mapping, relation_mask]:
                mapping.zero_()  # 置0处理
            relation_label.fill_(IGNORE_INDEX)  # 都填充上-100
            context_idxs.copy_(
                torch.from_numpy(self.data_train_token[docid, :, :]))
            if self.use_graph:
                graph_neib[i].copy_(torch.from_numpy(self.data_train_neib[docid, :, :]))
                graph_edge[i].copy_(torch.from_numpy(self.data_train_edge[docid, :, :]))

            ins = self.train_file[docid]
            cur_s_len = len(ins['sents'])
            cur_v_num = len(ins['vertexSet'])

            for vid, vertex in enumerate(ins['vertexSet']):
                sid = vertex['sent_id']
                pos = vertex['pos']
                label = vertex['label']

                # average representation of tokens in span
                vertex_mapping[vid, sid, pos[0]:pos[1]] = 1.0 / (pos[1] - pos[0])
                relation_label[vid] = label
                relation_mask[vid] = 1

            input_lengths = (context_idxs[:cur_s_len] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            yield {'context_idxs': context_idxs[:cur_s_len, :max_c_len].contiguous(),
                   'graph_neib': curr_graph_neib,
                   'graph_edge': curr_graph_edge,
                   'vertex_mapping': vertex_mapping[:cur_v_num, :cur_s_len, :max_c_len],
                   'relation_label': relation_label[:cur_v_num].contiguous(),
                   'input_lengths': input_lengths,
                   'relation_mask': relation_mask[:cur_v_num],
                   'dockey': ins['dockey'],  # dockeys in raw, used for bert
                   'index': docid,  # used for get bert representation, doc_ids in raw
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

            yield {'context_idxs': context_idxs[:cur_s_len, :max_c_len].contiguous(),
                   'graph_neib': curr_graph_neib,
                   'graph_edge': curr_graph_edge,
                   'vertex_mapping': vertex_mapping[:cur_v_num, :cur_s_len, :max_c_len],
                   'relation_label': relation_label[:cur_v_num].contiguous(),
                   'input_lengths': input_lengths,
                   'relation_mask': relation_mask[:cur_v_num],
                   'dockey': ins['dockey'],  # dockeys in raw, used for bert
                   'index': docid,  # used for get bert representation, doc_ids in raw
                   }

    def get_dev_batch(self):
        random.shuffle(self.dev_order)
        context_idxs = torch.LongTensor(self.dev_max_sent_num, self.max_length).cuda()

        vertex_mapping = torch.Tensor(self.max_vertex_num, self.dev_max_sent_num, self.max_length).cuda()
        relation_mask = torch.Tensor(self.max_vertex_num).cuda()  # ent pair in doc [max_vertex_num]
        relation_label = torch.LongTensor(self.max_vertex_num).cuda()  # relation label idx

        for i, docid in enumerate(self.dev_order):  # i: docid_in_iteration, docid: doc_index_in_rawData
            curr_graph_neib, curr_graph_edge = None, None
            for mapping in [vertex_mapping, relation_mask]:
                mapping.zero_()
            relation_label.fill_(IGNORE_INDEX)
            context_idxs.copy_(
                torch.from_numpy(self.data_dev_token[docid, :, :]))
            ins = self.dev_file[docid]
            cur_s_len = len(ins['sents'])
            cur_v_num = len(ins['vertexSet'])

            for vid, vertex in enumerate(ins['vertexSet']):
                sid = vertex['sent_id']
                pos = vertex['pos']
                label = vertex['label']

                # average representation of tokens in span
                vertex_mapping[vid, sid, pos[0]:pos[1]] = 1.0 / (pos[1] - pos[0])
                relation_label[vid] = label
                relation_mask[vid] = 1

            input_lengths = (context_idxs[:cur_s_len] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            yield {'context_idxs': context_idxs[:cur_s_len, :max_c_len].contiguous(),
                   'graph_neib': curr_graph_neib,
                   'graph_edge': curr_graph_edge,
                   'vertex_mapping': vertex_mapping[:cur_v_num, :cur_s_len, :max_c_len],
                   'relation_label': relation_label[:cur_v_num].contiguous(),
                   'input_lengths': input_lengths,
                   'relation_mask': relation_mask[:cur_v_num],
                   'dockey': ins['dockey'],  # dockeys in raw, used for bert
                   'index': docid,  # used for get bert representation, doc_ids in raw
                   }

    def train(self, model_pattern, model_name):  # model_pattern:BiLSTM model_name:checkpoint_BiLSTM
        print("Current training model will be saved to", model_name)
        params_arr = ["Hyper parameters: max_epoch:", self.max_epoch, "| optim:", self.opt_method, "lr:", self.lr,
                      "dropout:", 1 - self.keep_prob, "Category number:", self.relation_num,
                      "|| Model paramters: use_bert:", self.use_bert, "lstm_hd size:", self.lstm_hidden_size,
                      "| use_graph:", self.use_graph, "iter:", self.graph_iter, "graph_drop:", self.graph_drop,
                      "edge_num", self.max_edge_num, "ablate_edge:", self.ablation]
        logstr = " ------log:train=1648,test=400-----N/A/D/S---"
        if self.use_bert:
            logstr = logstr + "    " + self.bertRep.filename
        with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
            f_log.write("\n\n" + "------New train------" + time.asctime() + logstr + "\n")
            f_log.write(" ".join([str(item) for item in params_arr]) + "\n")
            print("------New train------" + time.asctime())
            print(logstr)
            print(params_arr)

        ori_model = model_pattern(config=self)
        if self.pretrain_model != None:
            ori_model.load_state_dict(torch.load(self.pretrain_model))
        ori_model.cuda()
        model = ori_model

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
        nll_average = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # 交叉熵损失，用于多分类

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        best_f1 = 0.0
        best_epoch = 0
        best_macro_f1 = 0.0
        best_macro_f1_epoch = 0
        best_micro_f1 = 0.0
        best_micro_f1_epoch = 0

        model.train()

        global_step = 0
        total_loss = 0
        start_time = time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log/", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        raw_model = model
        for epoch in range(self.max_epoch):
            model = raw_model
            model.train()
            self.acc_total.clear()
            batch_num = 0
            correct_money_num = 0

            if epoch - best_epoch > 5:
                self.lr *= 0.8
                print("-- changed lr ", self.lr, " at epoch ", epoch)
            if epoch - best_epoch >= 10:
                break  # early stop
            real_add1_num = 0
            real_deter2_num = 0
            real_sub3_num = 0
            real_none0_num = 0
            predicted_add1_num = 0
            predicted_deter2_num = 0
            predicted_sub3_num = 0
            predicted_none0_num = 0
            add1_TP = 0.0
            add1_FP = 0.0
            add1_FN = 0.0
            deter2_TP = 0.0
            deter2_FP = 0.0
            deter2_FN = 0.0
            sub3_TP = 0.0
            sub3_FP = 0.0
            sub3_FN = 0.0
            none0_TP = 0.0
            none0_FP = 0.0
            none0_FN = 0.0

            for data in self.get_train_batch():
                batch_num += 1
                context_idxs = data['context_idxs']
                graph_neib = data['graph_neib']  # None if not use_graph
                graph_edge = data['graph_edge']  # None if not use_graph
                vertex_mapping = data['vertex_mapping']
                relation_label = data['relation_label']
                input_lengths = data['input_lengths']
                relation_mask = data['relation_mask']
                dockey = data['dockey']
                index = data['index']

                sents_count = len(input_lengths)
                predict_re = model(context_idxs, input_lengths, vertex_mapping, graph_neib, graph_edge, dockey,sents_count)

                # CE loss
                loss = nll_average(predict_re.reshape(-1, self.relation_num), relation_label.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                output = torch.argmax(predict_re, dim=-1)

                instance = self.train_file[index]
                total_money = 0.0
                add_count = 0
                determine_count = 0
                deter_money_list = []
                instance_vertexSet = copy.deepcopy(instance['vertexSet'])
                for vid, vertex in enumerate(instance_vertexSet):
                    pred_label = output[vid].cuda().item()
                    if pred_label == 1:
                        add_count += 1
                        vertex['label'] = 1
                    elif pred_label == 2:  # D
                        determine_count += 1
                        deter_money_list.append(vertex['name'])
                        vertex['label'] = 2
                    elif pred_label == 3:  # S
                        vertex['label'] = 3
                    else:  # N
                        vertex['label'] = 0

                for raw, now in zip(instance['vertexSet'], instance_vertexSet):
                    if raw['label'] == 1 or now['label'] == 1:
                        if raw['label'] == 1:
                            real_add1_num += 1
                            if now['label'] == 1:
                                add1_TP += 1
                                predicted_add1_num += 1
                            else:
                                add1_FN += 1
                        else:
                            add1_FP += 1
                            predicted_add1_num += 1
                    if raw['label'] == 2 or now['label'] == 2:
                        if raw['label'] == 2:
                            real_deter2_num += 1
                            if now['label'] == 2:
                                deter2_TP += 1
                                predicted_deter2_num += 1
                            else:
                                deter2_FN += 1
                        else:
                            deter2_FP += 1
                            predicted_deter2_num += 1
                    if raw['label'] == 3 or now['label'] == 3:
                        if raw['label'] == 3:
                            real_sub3_num += 1
                            if now['label'] == 3:
                                sub3_TP += 1
                                predicted_sub3_num += 1
                            else:
                                sub3_FN += 1
                        else:
                            sub3_FP += 1
                            predicted_sub3_num += 1
                    if raw['label'] == 0 or now['label'] == 0:
                        if raw['label'] == 0:
                            real_none0_num += 1
                            if now['label'] == 0:
                                none0_TP += 1
                                predicted_none0_num += 1
                            else:
                                none0_FN += 1
                        else:
                            none0_FP += 1
                            predicted_none0_num += 1

                # accuracy
                for raw, now in zip(instance['vertexSet'], instance_vertexSet):
                    self.acc_total.add(raw['label'] == now['label'])

                if add_count + determine_count == 0:
                    for vertex in instance_vertexSet:
                        total_money += self.money_str2num(vertex['name'])
                elif determine_count == 1:
                    for vertex in instance_vertexSet:
                        if vertex['label'] == 2:
                            total_money += self.money_str2num(vertex['name'])
                            break
                elif add_count == 0 and determine_count != 0:
                    for vertex in instance_vertexSet:
                        if vertex['label'] == 2:
                            total_money += self.money_str2num(vertex['name'])
                elif len(deter_money_list) == 2 and deter_money_list[0] == deter_money_list[1]:
                    total_money += self.money_str2num(deter_money_list[0])
                else:
                    for vertex in instance_vertexSet:
                        if vertex['label'] == 1:
                            total_money += self.money_str2num(vertex['name'])
                        elif vertex['label'] == 3:
                            total_money -= self.money_str2num(vertex['name'])

                if (str(total_money)) == instance['total']:
                    correct_money_num += 1

                global_step += 1
                total_loss += loss.item()

            add_precision = add1_TP / (add1_TP + add1_FP) * 100.0
            add_recall = add1_TP / (add1_TP + add1_FN) * 100.0
            add_f1 = 2 * add_precision * add_recall / (add_precision + add_recall)
            deter_precision = 0
            if deter2_TP + deter2_FP != 0:
                deter_precision = deter2_TP / (deter2_TP + deter2_FP) * 100.0
            deter_recall = 0
            if deter2_TP+deter2_FN != 0:
                deter_recall = deter2_TP / (deter2_TP + deter2_FN) * 100.0
            deter_f1 = 0
            if deter_precision+deter_recall != 0:
                deter_f1 = 2 * deter_precision * deter_recall / (deter_precision + deter_recall)
            sub_precision = 0.0
            if sub3_TP + sub3_FP != 0:
                sub_precision = sub3_TP / (sub3_TP + sub3_FP) * 100.0
            sub_recall = 0.0
            if sub3_TP + sub3_FN != 0:
                sub_recall = sub3_TP / (sub3_TP + sub3_FN) * 100.0
            sub_f1 = 0.0
            if sub_precision + sub_recall != 0:
                sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall)
            none_precision = none0_TP / (none0_TP + none0_FP) * 100.0
            none_recall = none0_TP / (none0_TP + none0_FN) * 100.0
            none_f1 = 2 * none_precision * none_recall / (none_precision + none_recall)
            macro_precision = (add_precision + deter_precision + sub_precision + none_precision) / 4
            macro_recall = (add_recall + deter_recall + sub_recall + none_recall) / 4
            macro_f1 = (add_f1 + deter_f1 + sub_f1 + none_f1) / 4
            micro_precision = (add1_TP+deter2_TP+sub3_TP+none0_TP) / (add1_TP+deter2_TP+sub3_TP+none0_TP + add1_FP+deter2_FP+sub3_FP+none0_FP) * 100.0
            micro_recall = (add1_TP+deter2_TP+sub3_TP+none0_TP) / (add1_TP+deter2_TP+sub3_TP+none0_TP + add1_FN+deter2_FN+sub3_FN+none0_FN) * 100.0
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_macro_f1_epoch = epoch
            if micro_f1 > best_micro_f1:
                best_micro_f1 = micro_f1
                best_micro_f1_epoch = epoch

            use_f1 = "micro-f1"
            if micro_f1 > best_f1:
                best_f1 = micro_f1
                best_epoch = epoch
                path = os.path.join(self.checkpoint_dir, model_name)
                torch.save(ori_model.state_dict(), path)
                out_file_csv = "./log/result.csv"
                csvfile = open(out_file_csv, 'w', newline='', encoding='utf-8')
                writer = csv.writer(csvfile)
                result_list = [['max_epoch', 'best_epoch', 'use_f1', 'loss', 'macro-p', 'macro-r', 'macro-f',
                                'micro-p', 'micro-r', 'micro-f', 'label_acc', 'Money_p'],
                               [self.max_epoch, best_epoch, use_f1, round((total_loss / batch_num), 5),
                                round(macro_precision, 2), round(macro_recall, 2), round(macro_f1, 2),
                                round(micro_precision, 2), round(micro_recall, 2), round(micro_f1, 2),
                                round(self.acc_total.get() * 100.0, 2), round((correct_money_num * 100.0 / self.train_len), 2)],
                               ['', '', real_add1_num, 'add-1', round(add_precision, 2), round(add_recall, 2), round(add_f1, 2)],
                               ['', '', real_deter2_num, 'deter-2', round(deter_precision, 2), round(deter_recall, 2), round(deter_f1, 2)],
                               ['', '', real_sub3_num, 'sub-3', round(sub_precision, 2), round(sub_recall, 2), round(sub_f1, 2)],
                               ['', '', real_none0_num, 'none-0', round(none_precision, 2), round(none_recall, 2), round(none_f1, 2)]]
                writer.writerows(result_list)
                csvfile.close()
                out_file_csv1 = "./log/result1.csv"
                csvfile1 = open(out_file_csv1, 'w', newline='', encoding='utf-8')
                writer1 = csv.writer(csvfile1)
                result_list1 = [['test_money_acc', 'test_label_acc', '',
                                 'best epoch', 'loss', 'macro-p', 'macro-r', 'macro-f', 'micro-p', 'micro-r', 'micro-f',
                                 'label_acc', 'Money_acc', 'add-p', 'add-r', 'add-f', 'deter-p', 'deter-r', 'deter-f',
                                 'sub-p', 'sub-r', 'sub-f', 'none-p', 'none-r', 'none-f',
                                 '', 'loss', 'macro-p', 'macro-r', 'macro-f', 'micro-p', 'micro-r', 'micro-f',
                                 'label_acc', 'Money_acc', 'add-p', 'add-r', 'add-f', 'deter-p', 'deter-r', 'deter-f',
                                 'sub-p', 'sub-r', 'sub-f', 'none-p', 'none-r', 'none-f'],
                                ['', '', '', best_epoch, round((total_loss / batch_num), 5),
                                 round(macro_precision, 2), round(macro_recall, 2), round(macro_f1, 2),
                                 round(micro_precision, 2), round(micro_recall, 2), round(micro_f1, 2),
                                 round(self.acc_total.get() * 100.0, 2), round((correct_money_num * 100.0 / self.train_len), 2),
                                 round(add_precision, 2), round(add_recall, 2), round(add_f1, 2),
                                 round(deter_precision, 2), round(deter_recall, 2), round(deter_f1, 2),
                                 round(sub_precision, 2), round(sub_recall, 2), round(sub_f1, 2),
                                 round(none_precision, 2), round(none_recall, 2), round(none_f1, 2), '', '']]
                writer1.writerows(result_list1)
                csvfile1.close()

            logging(
                '| train epoch {:3d} | time: {:5.2f}s Loss {:5.5f} Macro-P {:5.2f} Macro-R {:5.2f} Macro-F1 {:5.2f} '
                'Micro-P {:5.2f} Micro-R {:5.2f} Micro-F1 {:5.2f} label_predict_acc {:5.2f} Money_acc {:5.2f}%'.format(
                    epoch, time.time() - start_time, total_loss / batch_num, macro_precision, macro_recall, macro_f1,
                    micro_precision, micro_recall, micro_f1, self.acc_total.get() * 100.0, correct_money_num * 100.0 / self.train_len))
            logging(
                '| add_precision {:5.2f} add_recall {:5.2f} add_f1 {:5.2f}  | deter_precision {:5.2f} deter_recall {:5.2f} deter_f1 {:5.2f}'
                    .format(add_precision, add_recall, add_f1, deter_precision, deter_recall, deter_f1))
            logging(
                '| sub_precision {:5.2f} sub_recall {:5.2f} sub_f1 {:5.2f}  | none_precision {:5.2f} none_recall {:5.2f} none_f1 {:5.2f}'
                    .format(sub_precision, sub_recall, sub_f1, none_precision, none_recall, none_f1))
            logging('|            real label num: Add {:5d}  Decisive {:5d}  Sub {:5d}  None {:5d}'
                    .format(real_add1_num, real_deter2_num, real_sub3_num, real_none0_num))
            logging('|       predicted label num: Add {:5d}  Decisive {:5d}  Sub {:5d}  None {:5d}'
                    .format(predicted_add1_num, predicted_deter2_num, predicted_sub3_num, predicted_none0_num))
            logging('| predicted right label num: Add {:5d}  Decisive {:5d}  Sub {:5d}  None {:5d}'
                    .format(int(add1_TP), int(deter2_TP), int(sub3_TP), int(none0_TP)))
            logging(
                '| best_macro_f1_epoch {:3d} with_F1 {:5.3f} | best_micro_f1_epoch {:3d} with F1 {:5.3f}% || now_use:{:},  best_epoch {:3d} with_F1 {:5.3f}'
                    .format(best_macro_f1_epoch, best_macro_f1, best_micro_f1_epoch, best_micro_f1, use_f1, best_epoch, best_f1))
            now_loss = total_loss/batch_num
            train_label_acc = self.acc_total.get() * 100.0
            total_loss = 0

            raw_model = model
            model = model_pattern(config=self)
            model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))
            model.cuda()
            model.eval()  # 测试过程中会使用model.eval()，这时神经网络会沿用batch normalization的值，并不使用drop out
            eval_result = self.eval(model, model_name, True, False)
            logging('-' * 89)
            start_time = time.time()


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

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        prefix = "test" if self.is_test else "dev"

        nll_average = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        wrong_predicted_list = []
        batch_num = 0
        correct_money_num = 0
        self.acc_total.clear()
        real_add1_num = 0
        real_deter2_num = 0
        real_sub3_num = 0
        real_none0_num = 0
        predicted_add1_num = 0
        predicted_deter2_num = 0
        predicted_sub3_num = 0
        predicted_none0_num = 0
        add1_TP = 0.0
        add1_FP = 0.0
        add1_FN = 0.0
        deter2_TP = 0.0
        deter2_FP = 0.0
        deter2_FN = 0.0
        sub3_TP = 0.0
        sub3_FP = 0.0
        sub3_FN = 0.0
        none0_TP = 0.0
        none0_FP = 0.0
        none0_FN = 0.0
        if output_analyze:
            analyze_file = open("log/analyze_" + prefix + "_" + model_name, "w")
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
                predict_re = model(context_idxs, input_lengths, vertex_mapping, graph_neib, graph_edge, dockey,
                                   sents_count)
                # ce loss
                loss = nll_average(predict_re.reshape(-1, self.relation_num), relation_label.reshape(-1))
                total_loss += loss.item()

                # max(prob) -> label
                output = torch.argmax(predict_re, dim=-1)

                if output_file:
                    instance = self.test_file[index]
                    sents = instance['sents']
                    if output_analyze:
                        analyze_file.write("\n" + "*" * 50 + str(index) + "\n")
                        analyze_file.write("Doc: " + instance['doc_text'] + "\n")

                    total_money = 0.0
                    add_count = 0
                    determine_count = 0
                    deter_money_list = []
                    self.test_file = json.load(
                        open(os.path.join(self.data_path, prefix + '.json'), encoding='utf-8'))  # prepro/train.json
                    instance = self.test_file[index]
                    for vid, vertex in enumerate(instance['vertexSet']):
                        try:
                            pred_label = output[vid].cuda().item()
                        except:
                            print("------")
                        if output_analyze:
                            analyze_file.write("Money"+str(vid)+":"+str(vertex["name"])+", label:"+str(vertex["label"])+", predicted:"+str(pred_label)+"\n")

                        if vertex['label'] == 1 or pred_label == 1:
                            if vertex['label'] == 1:
                                real_add1_num += 1
                                if pred_label == 1:
                                    add1_TP += 1
                                    predicted_add1_num += 1
                                else:
                                    add1_FN += 1
                            else:
                                add1_FP += 1
                                predicted_add1_num += 1
                        if vertex['label'] == 2 or pred_label == 2:
                            if vertex['label'] == 2:
                                real_deter2_num += 1
                                if pred_label == 2:
                                    deter2_TP += 1
                                    predicted_deter2_num += 1
                                else:
                                    deter2_FN += 1
                            else:
                                deter2_FP += 1
                                predicted_deter2_num += 1
                        if vertex['label'] == 3 or pred_label == 3:
                            if vertex['label'] == 3:
                                real_sub3_num += 1
                                if pred_label == 3:
                                    sub3_TP += 1
                                    predicted_sub3_num += 1
                                else:
                                    sub3_FN += 1
                            else:
                                sub3_FP += 1
                                predicted_sub3_num += 1
                        if vertex['label'] == 0 or pred_label == 0:
                            if vertex['label'] == 0:
                                real_none0_num += 1
                                if pred_label == 0:
                                    none0_TP += 1
                                    predicted_none0_num += 1
                                else:
                                    none0_FN += 1
                            else:
                                none0_FP += 1
                                predicted_none0_num += 1

                        # 计算accuracy
                        self.acc_total.add(pred_label == vertex['label'])  # 计算accuracy
                        if pred_label == 1:  # 就是label==A
                            add_count += 1
                            vertex['label'] = 1
                        elif pred_label == 2:
                            determine_count += 1
                            deter_money_list.append(vertex['name'])
                            vertex['label'] = 2
                        elif pred_label == 3:
                            vertex['label'] = 3
                        else:
                            vertex['label'] = 0

                    # 计算结果
                    if add_count + determine_count == 0:
                        for vertex in instance['vertexSet']:
                            total_money += self.money_str2num(vertex['name'])
                    elif determine_count == 1:
                        for vertex in instance['vertexSet']:
                            if vertex['label'] == 2:
                                total_money += self.money_str2num(vertex['name'])
                                break
                    elif add_count == 0 and determine_count != 0:
                        for vertex in instance['vertexSet']:
                            if vertex['label'] == 2:
                                total_money += self.money_str2num(vertex['name'])
                    elif len(deter_money_list) == 2 and deter_money_list[0] == deter_money_list[1]:
                        total_money += self.money_str2num(deter_money_list[0])
                    else:
                        for vertex in instance['vertexSet']:
                            if vertex['label'] == 1:
                                total_money += self.money_str2num(vertex['name'])
                            elif vertex['label'] == 3:
                                total_money -= self.money_str2num(vertex['name'])

                    if (str(total_money)) == instance['total']:
                        correct_money_num += 1
                    else:
                        wrong_predicted_list.append(index)
                    total_money_int_list.append(total_money)

        analyze_file.write("\n\nwrong predicted doc_index list\n")
        for indexx in wrong_predicted_list:
            analyze_file.write(str(indexx)+" ")
        analyze_file.close()
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
                pred_file.write(json.dumps(int_item, ensure_ascii=False) + "\n")

        add_precision = add1_TP / (add1_TP + add1_FP) * 100.0
        add_recall = add1_TP / (add1_TP + add1_FN) * 100.0
        add_f1 = 2 * add_precision * add_recall / (add_precision + add_recall)
        deter_precision = 0
        if deter2_TP + deter2_FP != 0:
            deter_precision = deter2_TP / (deter2_TP + deter2_FP) * 100.0
        deter_recall = 0
        if deter2_TP + deter2_FN != 0:
            deter_recall = deter2_TP / (deter2_TP + deter2_FN) * 100.0
        deter_f1 = 0
        if deter_precision + deter_recall != 0:
            deter_f1 = 2 * deter_precision * deter_recall / (deter_precision + deter_recall)
        sub_precision = 0.0
        if sub3_TP + sub3_FP != 0:
            sub_precision = sub3_TP / (sub3_TP + sub3_FP) * 100.0
        sub_recall = 0.0
        if sub3_TP + sub3_FN != 0:
            sub_recall = sub3_TP / (sub3_TP + sub3_FN) * 100.0
        sub_f1 = 0.0
        if sub_precision + sub_recall != 0:
            sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall)
        none_precision = none0_TP / (none0_TP + none0_FP) * 100.0
        none_recall = none0_TP / (none0_TP + none0_FN) * 100.0
        none_f1 = 2 * none_precision * none_recall / (none_precision + none_recall)
        macro_precision = (add_precision + deter_precision + sub_precision + none_precision) / 4
        macro_recall = (add_recall + deter_recall + sub_recall + none_recall) / 4
        macro_f1 = (add_f1 + deter_f1 + sub_f1 + none_f1) / 4
        micro_precision = (add1_TP + deter2_TP + sub3_TP + none0_TP) / (add1_TP + deter2_TP + sub3_TP + none0_TP + add1_FP + deter2_FP + sub3_FP + none0_FP) * 100.0
        micro_recall = (add1_TP + deter2_TP + sub3_TP + none0_TP) / (add1_TP + deter2_TP + sub3_TP + none0_TP + add1_FN + deter2_FN + sub3_FN + none0_FN) * 100.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

        out_file_csv = "./log/result.csv"
        csvfile = open(out_file_csv, 'a', newline='', encoding='utf-8')
        writer = csv.writer(csvfile)
        result_list = [['', '', '', '', '', '', '', '', '', ''],
                       ['', 'loss', 'macro-p', 'macro-r', 'macro-f', 'micro-p', 'micro-r', 'micro-f', 'label_acc', 'Money_acc'],
                       ['', round((total_loss / batch_num), 5),
                        round(macro_precision, 2), round(macro_recall, 2), round(macro_f1, 2),
                        round(micro_precision, 2), round(micro_recall, 2), round(micro_f1, 2),
                        round(self.acc_total.get() * 100.0, 2), round((correct_money_num * 100.0 / self.test_len), 2)],
                       [real_add1_num, 'add-1', round(add_precision, 2), round(add_recall, 2), round(add_f1, 2)],
                       [real_deter2_num, 'deter-2', round(deter_precision, 2), round(deter_recall, 2),round(deter_f1, 2)],
                       [real_sub3_num, 'sub-3', round(sub_precision, 2), round(sub_recall, 2), round(sub_f1, 2)],
                       [real_none0_num, 'none-0', round(none_precision, 2), round(none_recall, 2), round(none_f1, 2)]]
        writer.writerows(result_list)
        csvfile.close()
        out_file_csv1 = "./log/result1.csv"
        csvfile1 = open(out_file_csv1, 'a', newline='', encoding='utf-8')
        writer1 = csv.writer(csvfile1)
        result_list1 = [[round((correct_money_num * 100.0 / self.test_len), 2), round(self.acc_total.get() * 100.0, 2),
                         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                         round((total_loss / batch_num), 5),
                         round(macro_precision, 2), round(macro_recall, 2), round(macro_f1, 2),
                         round(micro_precision, 2), round(micro_recall, 2), round(micro_f1, 2),
                         round(self.acc_total.get() * 100.0, 2), round((correct_money_num * 100.0 / self.test_len), 2),
                         round(add_precision, 2), round(add_recall, 2), round(add_f1, 2),
                         round(deter_precision, 2), round(deter_recall, 2), round(deter_f1, 2),
                         round(sub_precision, 2), round(sub_recall, 2), round(sub_f1, 2),
                         round(none_precision, 2), round(none_recall, 2), round(none_f1, 2)]]
        writer1.writerows(result_list1)
        csvfile1.close()

        logging(
            "| eval {:5} | time: {:5.2f}s Loss{:8.3f} Macro-P {:5.2f} Macro-R {:5.2f} Macro-F1 {:5.2f} "
            "Micro-P {:5.2f} Micro-R {:5.2f} Micro-F1 {:5.2f} label_predict_acc {:5.2f} Money_Accuracy {:5.2f}%".format(
                prefix, time.time()-eval_start_time, total_loss / batch_num, macro_precision, macro_recall, macro_f1,
                    micro_precision, micro_recall, micro_f1, self.acc_total.get() * 100.0, correct_money_num * 100.0 / self.test_len))
        logging(
            '| add_precision {:5.2f} add_recall {:5.2f} add_f1 {:5.2f}  | deter_precision {:5.2f} deter_recall {:5.2f} deter_f1 {:5.2f}'
                .format(add_precision, add_recall, add_f1, deter_precision, deter_recall, deter_f1))
        logging(
            '| sub_precision {:5.2f} sub_recall {:5.2f} sub_f1 {:5.2f}  | none_precision {:5.2f} none_recall {:5.2f} none_f1 {:5.2f}'
                .format(sub_precision, sub_recall, sub_f1, none_precision, none_recall, none_f1))
        logging('|            real label num: Add {:5d}  Decisive {:5d}  Sub {:5d}  None {:5d}'
                .format(real_add1_num, real_deter2_num, real_sub3_num, real_none0_num))
        logging('|       predicted label num: Add {:5d}  Decisive {:5d}  Sub {:5d}  None {:5d}'
                .format(predicted_add1_num, predicted_deter2_num, predicted_sub3_num, predicted_none0_num))
        logging('| predicted right label num: Add {:5d}  Decisive {:5d}  Sub {:5d}  None {:5d}'
                .format(int(add1_TP), int(deter2_TP), int(sub3_TP), int(none0_TP)))
        logging('*' * 100)
        print("wrong_predicted_list:")
        print(wrong_predicted_list)
        return micro_f1

    def eval(self, model, model_name, output_file=False, output_analyze=False):
        total_money_int_list = []
        total_loss = 0
        start_time=time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        prefix = "dev"

        nll_average = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        batch_num = 0
        correct_money_num = 0
        self.acc_total.clear()
        real_add1_num = 0
        real_deter2_num = 0
        real_sub3_num = 0
        real_none0_num = 0
        predicted_add1_num = 0
        predicted_deter2_num = 0
        predicted_sub3_num = 0
        predicted_none0_num = 0
        add1_TP = 0.0
        add1_FP = 0.0
        add1_FN = 0.0
        deter2_TP = 0.0
        deter2_FP = 0.0
        deter2_FN = 0.0
        sub3_TP = 0.0
        sub3_FP = 0.0
        sub3_FN = 0.0
        none0_TP = 0.0
        none0_FP = 0.0
        none0_FN = 0.0
        for data in self.get_dev_batch():
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
                predict_re = model(context_idxs, input_lengths, vertex_mapping, graph_neib, graph_edge, dockey,
                                   sents_count)
                # ce loss
                loss = nll_average(predict_re.reshape(-1, self.relation_num), relation_label.reshape(-1))
                total_loss += loss.item()

                # max(prob) -> label
                output = torch.argmax(predict_re, dim=-1)

                if output_file:
                    instance = self.dev_file[index]
                    sents = instance['sents']
                    total_money = 0.0
                    add_count = 0
                    determine_count = 0
                    deter_money_list = []
                    self.dev_file = json.load(
                        open(os.path.join(self.data_path, prefix + '.json'), encoding='utf-8'))
                    instance = self.dev_file[index]
                    for vid, vertex in enumerate(instance['vertexSet']):
                        try:
                            pred_label = output[vid].cuda().item()
                        except:
                            print("------")

                        if vertex['label'] == 1 or pred_label == 1:
                            if vertex['label'] == 1:
                                real_add1_num += 1
                                if pred_label == 1:
                                    add1_TP += 1
                                    predicted_add1_num += 1
                                else:
                                    add1_FN += 1
                            else:
                                add1_FP += 1
                                predicted_add1_num += 1
                        if vertex['label'] == 2 or pred_label == 2:
                            if vertex['label'] == 2:
                                real_deter2_num += 1
                                if pred_label == 2:
                                    deter2_TP += 1
                                    predicted_deter2_num += 1
                                else:
                                    deter2_FN += 1
                            else:
                                deter2_FP += 1
                                predicted_deter2_num += 1
                        if vertex['label'] == 3 or pred_label == 3:
                            if vertex['label'] == 3:
                                real_sub3_num += 1
                                if pred_label == 3:
                                    sub3_TP += 1
                                    predicted_sub3_num += 1
                                else:
                                    sub3_FN += 1
                            else:
                                sub3_FP += 1
                                predicted_sub3_num += 1
                        if vertex['label'] == 0 or pred_label == 0:
                            if vertex['label'] == 0:
                                real_none0_num += 1
                                if pred_label == 0:
                                    none0_TP += 1
                                    predicted_none0_num += 1
                                else:
                                    none0_FN += 1
                            else:
                                none0_FP += 1
                                predicted_none0_num += 1

                        self.acc_total.add(pred_label == vertex['label'])
                        if pred_label == 1:
                            add_count += 1
                            vertex['label'] = 1
                        elif pred_label == 2:
                            determine_count += 1
                            deter_money_list.append(vertex['name'])
                            vertex['label'] = 2
                        elif pred_label == 3:
                            vertex['label'] = 3
                        else:
                            vertex['label'] = 0

                    if add_count + determine_count == 0:
                        for vertex in instance['vertexSet']:
                            total_money += self.money_str2num(vertex['name'])
                    elif determine_count == 1:
                        for vertex in instance['vertexSet']:
                            if vertex['label'] == 2:
                                total_money += self.money_str2num(vertex['name'])
                                break
                    elif add_count == 0 and determine_count != 0:
                        for vertex in instance['vertexSet']:
                            if vertex['label'] == 2:
                                total_money += self.money_str2num(vertex['name'])
                    elif len(deter_money_list) == 2 and deter_money_list[0] == deter_money_list[1]:
                        total_money += self.money_str2num(deter_money_list[0])
                    else:
                        for vertex in instance['vertexSet']:
                            if vertex['label'] == 1:
                                total_money += self.money_str2num(vertex['name'])
                            elif vertex['label'] == 3:
                                total_money -= self.money_str2num(vertex['name'])

                    if (str(total_money)) == instance['total']:
                        correct_money_num += 1
                    total_money_int_list.append(total_money)  # int_money_list

        add_precision = add1_TP / (add1_TP + add1_FP) * 100.0
        add_recall = add1_TP / (add1_TP + add1_FN) * 100.0
        add_f1 = 2 * add_precision * add_recall / (add_precision + add_recall)
        deter_precision = 0
        if deter2_TP + deter2_FP != 0:
            deter_precision = deter2_TP / (deter2_TP + deter2_FP) * 100.0
        deter_recall = 0
        if deter2_TP + deter2_FN != 0:
            deter_recall = deter2_TP / (deter2_TP + deter2_FN) * 100.0
        deter_f1 = 0
        if deter_precision + deter_recall != 0:
            deter_f1 = 2 * deter_precision * deter_recall / (deter_precision + deter_recall)
        sub_precision = 0.0
        if sub3_TP + sub3_FP != 0:
            sub_precision = sub3_TP / (sub3_TP + sub3_FP) * 100.0
        sub_recall = 0.0
        if sub3_TP + sub3_FN != 0:
            sub_recall = sub3_TP / (sub3_TP + sub3_FN) * 100.0
        sub_f1 = 0.0
        if sub_precision + sub_recall != 0:
            sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall)
        none_precision = none0_TP / (none0_TP + none0_FP) * 100.0
        none_recall = none0_TP / (none0_TP + none0_FN) * 100.0
        none_f1 = 2 * none_precision * none_recall / (none_precision + none_recall)
        macro_precision = (add_precision + deter_precision + sub_precision + none_precision) / 4
        macro_recall = (add_recall + deter_recall + sub_recall + none_recall) / 4
        macro_f1 = (add_f1 + deter_f1 + sub_f1 + none_f1) / 4
        micro_precision = (add1_TP + deter2_TP + sub3_TP + none0_TP) / (add1_TP + deter2_TP + sub3_TP + none0_TP + add1_FP + deter2_FP + sub3_FP + none0_FP) * 100.0
        micro_recall = (add1_TP + deter2_TP + sub3_TP + none0_TP) / (add1_TP + deter2_TP + sub3_TP + none0_TP + add1_FN + deter2_FN + sub3_FN + none0_FN) * 100.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

        logging(
            "| now_dev_eval | cost{:6.2f}s Loss{:8.3f} label_predict_acc {:5.2f} Money_accuracy {:5.2f}%".format(time.time()-start_time,
                total_loss / batch_num, self.acc_total.get() * 100.0, correct_money_num * 100.0 / self.dev_len))
        my_result = [total_loss / batch_num, macro_f1, micro_f1,
                     self.acc_total.get() * 100.0, correct_money_num * 100.0 / self.dev_len]
        return my_result

    def testall(self, model_pattern, model_name):
        print("Evaluating model from", model_name, time.asctime())
        params_arr = ["Hyper parameters: dropout:", 1 - self.keep_prob, " Category number:", self.relation_num,
                      "|| Model paramters: use_bert:", self.use_bert, "lstm_hd size:", self.lstm_hidden_size,
                      "| use_graph:", self.use_graph, "iter:", self.graph_iter, "graph_drop:", self.graph_drop,
                      "edge_num", self.max_edge_num, "ablate_edge:", self.ablation]
        print(params_arr)
        model = model_pattern(config=self)

        model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))
        model.cuda()
        model.eval()
        self.test(model, model_name, True, True)
