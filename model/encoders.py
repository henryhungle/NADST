import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time 
from torch.autograd import Variable
from collections import OrderedDict 
import pdb  
import json 
from utils.config import * 
from model.modules import *

class Encoder(nn.Module):
    def __init__(self, size, nb_layers):
        super(Encoder, self).__init__()
        self.norm = nn.ModuleList()
        self.nb_layers = nb_layers
        for n in range(nb_layers):
            self.norm.append(LayerNorm(size))

    def forward(self, *seqs):
        output = []
        i=0
        seq_i=0
        while(True):
            if seqs[seq_i] is None:
                output.append(None)
                seq_i+=1
                continue
            if isinstance(seqs[seq_i],list):
                output_seq = []
                for seq in seqs[seq_i]:
                    output_seq.append(self.norm[i](seq))
                    i+=1
                output.append(output_seq)
                seq_i+=1
            else:
                output.append(self.norm[i](seqs[seq_i]))
                i+=1
                seq_i+=1
            if i==self.nb_layers:
                break
        return output

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.zeros((x.shape[0], x.shape[1], self.d_model)).cuda()
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
