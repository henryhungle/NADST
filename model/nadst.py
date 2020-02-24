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
from model.generators import *
from model.encoders import * 
from model.fertility_decoder import * 
from model.state_decoder import * 
from model.evaluator import * 
from model.optimizer import * 
from model.predict import * 

class NADST(nn.Module):
    def __init__(self, encoder, encoder2, 
            in_embed, 
            domain_embed, domain_embed2,
            slot_embed, slot_embed2,
            fert_dec_nn, 
            lenval_gen, gate_gen, 
            state_dec_nn,
            state_gen, args):
        super(NADST, self).__init__()

        self.fert_decoder = Fertility_Decoder(encoder,
             fert_dec_nn, lenval_gen, gate_gen,
             in_embed, domain_embed, slot_embed, args)
        self.state_decoder = State_Decoder(encoder2,
             state_dec_nn, state_gen,
             domain_embed2, slot_embed2, args, self.fert_decoder)
        self.args = args
        
    def forward(self, b):
        out = self.fert_decoder(b) 
        out = self.state_decoder(b, out) 
        return out

def make_model(
        src_lang, tgt_lang,
        domain_lang, slot_lang,  
        len_val, 
        args):
    
    c = copy.deepcopy
    d_ff = args['d_model']*4
    src_vocab = src_lang.n_words
    tgt_vocab = tgt_lang.n_words
    slot_vocab = slot_lang.n_words
    domain_vocab = domain_lang.n_words
   
    att = MultiHeadedAttention(args['h_attn'], args['d_model'], dropout=args['drop'])
    ff = PositionwiseFeedForward(args['d_model'], d_ff, dropout=args['drop'])
    position = PositionalEncoding(args['d_model'], dropout=args['drop'])
    
    # Text embedding 
    if args['d_embed']!=-1:
        in_embed = [Embeddings(args['d_embed'], src_vocab), nn.Linear(args['d_embed'], args['d_model']), nn.ReLU(), c(position)]
    else:
        in_embed = [Embeddings(args['d_model'], src_vocab), c(position)]
    in_embed = nn.Sequential(*in_embed)

    # domain/slot embedding in fertility decoder 
    domain_embed =  None
    slot_embed = None
    # domain/slot embedding in state decoder 
    domain_embed2 =  None
    slot_embed2 = None
    if args['sep_input_embedding']:
        if args['no_pe_ds_emb1']:
            domain_embed = [Embeddings(args['d_model'], domain_vocab)]
            slot_embed = [Embeddings(args['d_model'], slot_vocab)]
        else:
            domain_embed = [Embeddings(args['d_model'], domain_vocab), c(position)]
            slot_embed = [Embeddings(args['d_model'], slot_vocab), c(position)]
        domain_embed = nn.Sequential(*domain_embed)
        slot_embed = nn.Sequential(*slot_embed)
        if not args['no_pe_ds_emb1'] and args['no_pe_ds_emb2']:
            domain_embed2 = domain_embed[0]
            slot_embed2 = slot_embed[0]
        elif args['no_pe_ds_emb1'] and not args['no_pe_ds_emb2']:
            domain_embed2 = [domain_embed[0], c(position)]
            domain_embed2 = nn.Sequential(*domain_embed2)
            slot_embed2 = [slot_embed[0], c(position)]
            slot_embed2 = nn.Sequential(*out_embed2)
    
    enc_layers=2
    if args['delex_his']:
        enc_layers += 1
    encoder = Encoder(args['d_model'], nb_layers=enc_layers)
    encoder2 = c(encoder) 
    
    gate_gen = None 
    nb_attn=2
    if args['delex_his']:
        nb_attn+=1
    fert_dec_layer = SubLayer(args['d_model'], c(att), c(ff), args['drop'], nb_attn=nb_attn) 
    fert_dec_nn = AttentionNet(fert_dec_layer, args['fert_dec_N'])
    lenval_gen = Generator(args['d_model'], len_val+1) #lenval+1 to include 0-length 
    if args['slot_gating']:
        gate_gen = Generator(args['d_model'], len(GATES))
    state_dec_layer = SubLayer(args['d_model'], c(att), c(ff), args['drop'], nb_attn=nb_attn)
    state_dec_nn = AttentionNet(state_dec_layer, args['state_dec_N'])
    if args['pointer_decoder']:
        pointer_attn = MultiHeadedAttention(1, args['d_model'], dropout=0.0)
        if args['sep_output_embedding']:
            state_gen = PointerGenerator(Generator(args['d_model'], src_vocab), pointer_attn, args)
        else:
            state_gen = PointerGenerator(Generator(args['d_model'], src_vocab, in_embed[0].lut.weight), pointer_attn, args)
    else:
        state_gen = PointerGenerator(Generator(args['d_model'], tgt_vocab), None, args)    

    model = NADST(
        encoder=encoder,
        encoder2=encoder2,
        in_embed=in_embed,
        domain_embed=domain_embed,
        domain_embed2=domain_embed2,
        slot_embed=slot_embed,
        slot_embed2=slot_embed2,
        fert_dec_nn=fert_dec_nn,
        lenval_gen=lenval_gen,
        gate_gen = gate_gen, 
        state_dec_nn=state_dec_nn,
        state_gen = state_gen,
        args = args
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

