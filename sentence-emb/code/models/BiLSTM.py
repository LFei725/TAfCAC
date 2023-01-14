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


class Decoder(nn.Module):
    def __init__(self, hidden_size, relation_num):
        super(Decoder, self).__init__()
        represent_size = hidden_size

        self.bank_size = 200
        self.activation = nn.ReLU()
        self.predict = nn.Sequential(
                nn.Linear(represent_size, self.bank_size),
                self.activation,
                nn.Dropout(0.2),
                nn.Linear(self.bank_size, relation_num)
        )

    def forward(self, context_output, vertex_mapping):
        money_num = vertex_mapping.shape[0]
        dim_size = context_output.shape[-1]
        vertex_mapping_r = vertex_mapping.reshape(money_num, -1)
        context_output_r = context_output.reshape(-1, dim_size)
        vertex_output = torch.matmul(vertex_mapping_r, context_output_r)
        predict_re = self.predict(vertex_output)
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
        self.iters = iters
        self.block = nn.ModuleList([GraphConvolutionLayer(max_edge,input_size,hidden_size,graph_drop) for i in range(iters)])

    def forward(self, nodes_embed,node_nei,node_edge):

        hi = nodes_embed
        for cnt in range(0, self.iters):
            hi = self.block[cnt](hi,node_nei,node_edge)
            nodes_embed = torch.cat((nodes_embed,hi),dim=-1)

        return nodes_embed


class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.config = config

        if self.config.use_bert:  # if we use precal BERT, hidden_size=768
            hidden_size = 768
            input_size = 768
            self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, True, 1 - config.keep_prob, False)
            self.linear_re = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.token_emb = nn.Embedding(config.data_token_vec.shape[0], config.data_token_vec.shape[1])
            self.token_emb.weight.data.copy_(torch.from_numpy(config.data_token_vec))
            input_size = config.data_token_vec.shape[1]
            hidden_size = config.lstm_hidden_size
            self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, True, 1 - config.keep_prob, False)
            self.linear_re = nn.Linear(hidden_size*2, hidden_size)

        self.repdrop = torch.nn.Dropout(p=1-config.keep_prob)
        # init graph network here
        if self.config.use_graph:
            self.graph_reason = GraphReasonLayer(config.max_edge_num,hidden_size,hidden_size, config.graph_iter, config.graph_drop)
            self.graph_size = (1+config.graph_iter)*hidden_size
            hidden_size = self.graph_size
        self.decoder = Decoder(hidden_size, config.relation_num)

    def getBertRep(self, doc_indexes, sents_num):
        reps = rnn.pad_sequence([torch.tensor(self.config.bertRep[doc_indexes + str(index)][()]) for index in range(sents_num)], batch_first=True)
        return reps.cuda()

    def forward(self, context_idxs, context_lens, vertex_mapping, graph_neib, graph_edge, doc_index, sents_num):
        if self.config.use_bert:  # use precaled bert
            bert_tensor = self.getBertRep(doc_index, sents_num)
            context_output = bert_tensor
            # new add
            context_output = self.rnn(context_output, context_lens)
            context_output = torch.relu(self.linear_re(context_output))
        else:  # use lstm
            sent = self.token_emb(context_idxs)
            context_output = self.rnn(sent, context_lens)
            context_output = torch.relu(self.linear_re(context_output))
        context_output = self.repdrop(context_output)

        # add graph network here
        if self.config.use_graph:
            N_word = context_output.shape[1]
            graph_out = self.graph_reason(context_output,graph_neib[:,:N_word,:],graph_edge[:,:N_word,:])
            predict_re = self.decoder(graph_out, vertex_mapping)
        else:
            predict_re = self.decoder(context_output, vertex_mapping)
        return predict_re


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EncoderRNN(nn.Module):
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
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])  # [2,1,num_units]
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, hidden = self.rnns[i](output, hidden)


            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


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


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)
