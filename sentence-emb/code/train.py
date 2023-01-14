import config
import models
import numpy as np
import os
import time
import datetime
import json
import sys
import os
import argparse

parser = argparse.ArgumentParser()
# common parameters
parser.add_argument('--model_name', type=str, default='BiLSTM', help='encoder model, BiLSTM/CNN3')
parser.add_argument('--save_name', type=str, default='checkpoint_BiLSTM', help='model save/log filename')
parser.add_argument('--eval', action="store_true")
parser.add_argument('--train_prefix', type=str, default='train')
parser.add_argument('--test_prefix', type=str, default='test')
parser.add_argument('--dev_prefix', type=str, default='dev')
parser.add_argument('--pretrain', action="store_true", help="continue training with pretrained model")
# training hyper-parameters
parser.add_argument('--max_epoch', type=int, default=15)
parser.add_argument('--test_epoch', type=int, default=1)
# parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=0.0005)  # bert 0.0002
# parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--dropout', type=float, default=0.3, help="dropout rate for bilstm/bert")
# model_configuration
parser.add_argument('--use_bert', action="store_true", help='use pre-cached bert')
parser.add_argument('--lstm_hidden_size', type=int, default=128)
parser.add_argument('--use_graph', action="store_true", help='use graph module')
parser.add_argument('--graph_drop', type=float, default=0.7, help="dropout rate for graph model")
parser.add_argument('--graph_iter', type=int, default=2, help="iteration number of GNN")
parser.add_argument('--ablation', type=int, default=-1)

args = parser.parse_args()
model = {
    'CNN3': models.CNN3,
    'LSTM': models.LSTM,
    'BiLSTM': models.BiLSTM,
    'ContextAware': models.ContextAware,
}

con = config.Config(args)

# training hyper-parameters
con.set_max_epoch(args.max_epoch)
con.set_test_epoch(args.test_epoch)
con.set_drop_prob(args.dropout)
con.set_lr(args.lr)
# model_configuration
con.set_lstm_hidden_size(args.lstm_hidden_size)
con.set_use_bert(args.use_bert)
con.set_use_graph(args.use_graph)
con.set_graph_iter(args.graph_iter)
con.set_graph_drop(args.graph_drop)
con.set_ablation(args.ablation)

if args.pretrain:
    con.set_pretrain_model("checkpoint/" + args.save_name)
    print("Initiate model with checkpoint/" + args.save_name, time.asctime())
con.load_other_data()
con.load_test_data()
con.load_dev_data()
if not args.eval:
    con.load_train_data()
    con.train(model[args.model_name], args.save_name)

con.testall(model[args.model_name], args.save_name)
