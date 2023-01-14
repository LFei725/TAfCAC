import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
from transformers import *

class Decoder(nn.Module):
    def __init__(self, hidden_size, relation_num, use_distance, dis_size):
        super(Decoder, self).__init__()
        self.use_distance = use_distance
        represent_size = hidden_size
        if self.use_distance:
            represent_size += dis_size
            self.dis_embed = nn.Embedding(20, dis_size, padding_idx=10)

        self.use_bilinear = False
        if self.use_bilinear:
            self.predict = torch.nn.Bilinear(represent_size, represent_size, relation_num)
        else:
            self.bank_size = 200
            self.activation = nn.ReLU()
            self.predict = nn.Sequential(
                    nn.Linear(represent_size*2, self.bank_size*2),
                    self.activation,
                    nn.Dropout(0.2),
                    nn.Linear(self.bank_size*2, relation_num)
            )

    # context_output is encoder's output, with [batch_size, max_doc_length, hidden_size]
    def forward(self, context_output, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h):
        start_re_output = torch.matmul(h_mapping, context_output)
        end_re_output = torch.matmul(t_mapping, context_output)

        if self.use_distance:
            s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
            t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
            if self.use_bilinear:
                predict_re = self.predict(s_rep, t_rep)
            else:
                predict_re = self.predict(torch.cat([s_rep, t_rep], dim=-1))
        else:
            if self.use_bilinear:
                predict_re = self.predict(start_re_output, end_re_output)
            else:
                predict_re = self.predict(torch.cat([start_re_output, end_re_output], dim=-1))

        return predict_re

class GraphConvolutionLayer(nn.Module):
    def __init__(self,max_edge,input_size,hidden_size,graph_drop):
        super(GraphConvolutionLayer, self).__init__()
        self.max_edge = max_edge
        self.W = nn.Parameter(torch.Tensor(size=(input_size, hidden_size)))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        self.W_edge = nn.ModuleList([nn.Linear(hidden_size,hidden_size,bias=False) for i in range(self.max_edge)])
        for m in self.W_edge:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.zeros_(self.bias)
        self.drop = torch.nn.Dropout(p=graph_drop, inplace=False)

    def forward(self, nodes_embed,node_nei,node_edge):

        N_bt = nodes_embed.shape[0]
        h = torch.matmul(nodes_embed,self.W.unsqueeze(0))
        edge_embed_list = []
        for edge_type in range(self.max_edge):
            edge_embed_list.append(self.W_edge[edge_type](h).unsqueeze(1))
        edge_embed = torch.cat(edge_embed_list,dim=1)
        sum_nei = torch.zeros_like(h)
        for cnt in range(node_nei.shape[-1]):
            mask = (node_nei[...,cnt]>=0)
            batch_ind,word_ind = torch.where(mask)
            if batch_ind.shape[0] == 0:
                continue
            sum_nei[batch_ind,word_ind] += edge_embed[batch_ind,node_edge[...,cnt][batch_ind,word_ind],node_nei[...,cnt][batch_ind,word_ind]]
        degs = torch.sum(node_edge>=0,dim=-1).float().clamp(min=1).unsqueeze(dim=-1)
        norm = 1.0 / degs
        dst = sum_nei*norm + self.bias
        out = self.drop(torch.relu(dst))
        return out

class GraphReasonLayer(nn.Module):
    def __init__(self,max_edge,input_size,hidden_size,iters,graph_drop=0.0):
        super(GraphReasonLayer, self).__init__()
        # self.W = nn.Parameter(nn.init.normal_(torch.empty(input_size, input_size)), requires_grad=True)
        # self.W_node = nn.ModuleList([nn.Linear(input_size,input_size) for i in range(iters)])
        # self.W_sum = nn.ModuleList([nn.Linear(input_size,input_size) for i in range(iters)])
        self.iters = iters
        self.block = nn.ModuleList([GraphConvolutionLayer(max_edge,input_size,hidden_size,graph_drop) for i in range(iters)])

    def forward(self, nodes_embed,node_nei,node_edge):

        hi = nodes_embed
        for cnt in range(0, self.iters):
            hi = self.block[cnt](hi,node_nei,node_edge)
            nodes_embed = torch.cat((nodes_embed,hi),dim=-1)

        return nodes_embed

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config

        self.repdrop = torch.nn.Dropout(p=1-config.keep_prob)
        self.bert = BertModel.from_pretrained('bert-base-cased')
        input_size = 768

        if self.config.use_lstm:
            hidden_size = config.lstm_hidden_size
            self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, True, 1 - config.keep_prob, False)
            self.linear_re = nn.Linear(hidden_size*2, hidden_size)
        else:
            hidden_size = input_size

        # init graph network here
        if self.config.use_graph:
            self.graph_reason = GraphReasonLayer(config.max_edge_num,hidden_size,hidden_size, config.graph_iter, config.graph_drop)
            self.graph_size = (1+config.graph_iter)*hidden_size
            hidden_size = self.graph_size

        self.decoder = Decoder(hidden_size, config.relation_num, config.use_distance, config.dis_size)

    # input_ids: (batch_size, max_subtok_length)
    # word_mapping: batch_size, max_sent_len, max_subtok_length
    # rep: (sent_len, 768)
    # reps: (batch_size, max_sent_len, 768)
    def getBertRep(self, input_ids, word_mapping):
        #print(input_ids.shape)
        bs, subtok_num = input_ids.shape
        model_output = self.bert(input_ids, output_hidden_states=True)
        subtoken_reps = torch.mean(torch.stack(model_output.hidden_states[8:13]), 0) # layer 9, 10, 11, 12
        #print(subtoken_reps.shape)
        # subtoken_reps: (batch_size, #subtoken, 768)
        #print(word_mapping.shape)
        bert_output = torch.matmul(word_mapping, subtoken_reps)
        #print(bert_output.shape)
        return bert_output

    # graph_neib: [batch_size, max_node_num=512, max_neib_num]
    # max_node_num means maximum 512 words
    # max_neib_num is 20 [self-node, adjacent sentence, previous word, next word, coref main word, head words of dependency arcs ...]
    # doc_indexes: dockeys(filename in raw data file) in current batch
    def forward(self, context_idxs, context_lens, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, graph_neib, graph_edge, doc_indexes, bert_input_ids, bert_word_mapping):
        bert_output = self.getBertRep(bert_input_ids, bert_word_mapping) # batch_size, max_len, 768
        #if torch.sum(torch.isnan(bert_output))!=0:
        #    print(torch.isnan(bert_output))
        #    print(bert_output)
        assert torch.sum(torch.isnan(bert_output))==0
        if self.config.use_lstm:
            context_output = self.rnn(bert_output, context_lens)
            context_output = torch.relu(self.linear_re(context_output))
        else:
            context_output = self.repdrop(bert_output)

        # add graph network here
        if self.config.use_graph:
            N_word = context_output.shape[1]
            graph_out = self.graph_reason(context_output,graph_neib[:,:N_word,:],graph_edge[:,:N_word,:])
            predict_re = self.decoder(graph_out, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h)
        else:
            predict_re = self.decoder(context_output, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h)

        return predict_re

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, hidden = self.rnns[i](output, (hidden, c))


            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]
