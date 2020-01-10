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

class NBT(nn.Module):
    def __init__(self, encoder0, encoder1, encoder2,  
            in_embed, in_embed2, 
            out_embed, out_embed2,
            out_domain_embed, out_domain2_embed,
            context_nn, domain2slot_nn, slot2lenval_nn,
            lenval_gen, gate_gen, 
            tag_nn, tag_gen, tag_gen2,
            slot2state_nn, state_gen, args):
        super(NBT, self).__init__()

        self.fert_decoder = Fertility_Decoder(encoder0, encoder1,
             context_nn, domain2slot_nn, slot2lenval_nn,
             lenval_gen, gate_gen,
             in_embed, out_embed, out_domain_embed)
        self.state_decoder = State_Decoder(encoder2,
             slot2state_nn,
             in_embed2, out_embed2, out_domain2_embed, 
             state_gen)

        self.args = args
        
    def forward(self, b):
        out = self.decode1(b) 
        out = self.decode2(b, out) 
        return out
    
    def decode1(self, b):
        out = {}
        in_domainslots, delex_context, context, user_uttr, prev_bs = self.get_embedded0(b)
        out['encoded_context'] = context
        out['encoded_delex_context'] = delex_context
        out['encoded_user_uttr'] = user_uttr
        out['encoded_prev_bs'] = prev_bs 
        out['encoded_in_domainslots'] = in_domainslots
        out_slots = self.generate_slot_logits(b, prev_bs, delex_context, context, user_uttr, in_domainslots)
        out['out_slots'] = out_slots
        return out 

    def decode2(self, b, out):
        if 'auto_regressive' in self.args and self.args['auto_regressive']:
            y_in, delex_context2, context2 = self.get_embedded3(b, out) 
            out_states, context2 = self.generate_state_logits_atrg(b, out, y_in, delex_context2, context2)
            out['out_states'] = out_states
            out['encoded_context2'] = context2
            out['encoded_in_domainslots2'] = y_in
            return out 
        else:
            in_domainslots2, delex_context2, context2, user_uttr2, prev_bs2 = self.get_embedded2(b, out['encoded_delex_context'], out['encoded_context'], out['encoded_user_uttr'], out['encoded_prev_bs'], out['out_slots'])
            out_states = self.generate_state_logits(b, out['out_slots'], prev_bs2, delex_context2, context2, user_uttr2, in_domainslots2)
            out['out_states'] = out_states
            if self.args['pointer_decoder']:
                if not self.args['pointer_attn']: 
                    out['pointer_attn'] = self.slot2state_nn.layers[-1].attn[-2].attn.max(dim=1)[0]
                out['encoded_context2'] = context2
                out['encoded_in_domainslots2'] = in_domainslots2
        return out 
    
    def get_embedded0(self, b):
        user_uttr = None 
        prev_bs = None
        if self.args['delex_his']:
            delex_context = self.fert_decoder.in_embed(b['delex_context'])
        else:
            delex_context = None 
        context = self.fert_decoder.in_embed(b['context'])
        if 'context_nn_N' in self.args and self.args['context_nn_N'] > 0: 
            context = self.fert_decoder.context_nn(context, b['context_mask'])
        if self.args['sep_dialog_history']:
            user_uttr = self.fert_decoder.in_embed(b['user_uttr'])
        if self.args['previous_belief_state']:
            prev_bs = self.fert_decoder.in_embed(b['prev_bs'])
        if not self.args['sep_input_embedding']:
            in_domains = self.fert_decoder.in_embed(b['sorted_in_domains'])
            in_slots = self.fert_decoder.in_embed(b['sorted_in_slots'])
        else:
            in_domains = self.fert_decoder.out_domain_embed(b['sorted_in_domains'])
            in_slots = self.fert_decoder.out_embed(b['sorted_in_slots'])
        in_domainslots = in_domains + in_slots 
        return in_domainslots, delex_context, context, user_uttr, prev_bs
    
    def get_embedded3(self, b, encoded): #delex_context, context, out_slots):
        delex_context, context = encoded['encoded_delex_context'], encoded['encoded_context']          
        if self.args['sep_context_embedding']:
            context2 = self.state_decoder.in_embed2(b['context'])
            delex_context2 = self.state_decoder.in_embed2(b['delex_context'])
        else:
            delex_context2 = delex_context
            context2 = context 
        y_in = self.fert_decoder.in_embed(b['y_in'])
        if self.args['out2in_atrg']:
            out_slots = encoded['out_slots']
        else:
            out_slots = encoded['encoded_in_domainslots']
        context_vec = out_slots.reshape(-1, out_slots.shape[-1])
        context_vec = context_vec[b['sorted_in_domainslots2_idx']]
        y_in += context_vec.unsqueeze(1)
        return y_in, delex_context2, context2
    
    def get_embedded2(self, b, delex_context, context, user_uttr, prev_bs, out_slots):
        user_uttr2 = None 
        prev_bs2 = None 
        if 'auto_regressive' in self.args and self.args['auto_regressive']:
            context2 = self.state_decoder.in_embed2(b['context'])
            delex_context2 = self.state_decoder.in_embed2(b['delex_context'])
            y_in = self.state_decoder.in_embed2(b['y_in'])
            return y_in, delex_context2, context2, user_uttr2, prev_bs2
        else:
            if self.args['sep_context_embedding']:
                context2 = self.state_decoder.in_embed2(b['context'])
                delex_context2 = self.state_decoder.in_embed2(b['delex_context'])
                if self.args['sep_dialog_history']:
                    user_uttr2 = self.state_decoder.in_embed2(b['user_uttr'])
                if self.args['previous_belief_state']:
                    prev_bs2 = self.state_decoder.in_embed2(b['prev_bs'])
            else:
                delex_context2 = delex_context
                context2 = context 
                if self.args['sep_dialog_history']:
                    user_uttr2 = user_uttr
                if self.args['previous_belief_state']: 
                    prev_bs2 = prev_bs

            if not self.args['sep_input_embedding']:
                if self.args['sep_context_embedding']:
                    in_domains2 = self.state_decoder.in_embed2(b['sorted_in_domains2'])
                    in_slots2 = self.state_decoder.in_embed2(b['sorted_in_slots2'])
                else:
                    in_domains2 = self.fert_decoder.in_embed(b['sorted_in_domains2'])
                    in_slots2 = self.fert_decoder.in_embed(b['sorted_in_slots2'])
            else:
                if self.args['sep_embedding']:
                    in_domains2 = self.state_decoder.out_domain2_embed(b['sorted_in_domains2'])
                    in_slots2 = self.state_decoder.out_embed2(b['sorted_in_slots2'])
                else:
                    if 'no_pe_ds_emb1' in self.args and 'no_pe_ds_emb2' in self.args and \
                        ((not self.args['no_pe_ds_emb1'] and self.args['no_pe_ds_emb2']) or \
                        (self.args['no_pe_ds_emb1'] and not self.args['no_pe_ds_emb2'])):
                        in_domains2 = self.state_decoder.out_domain2_embed(b['sorted_in_domains2'])
                        in_slots2 = self.state_decoder.out_embed2(b['sorted_in_slots2'])
                    else:
                        in_domains2 = self.fert_decoder.out_domain_embed(b['sorted_in_domains2'])
                        in_slots2 = self.fert_decoder.out_embed(b['sorted_in_slots2'])         
            in_domainslots2 = in_domains2 + in_slots2
            return in_domainslots2, delex_context2, context2, user_uttr2, prev_bs2
        
    def generate_slot_logits(self, b, prev_bs, delex_context, context, user_uttr, in_domainslots):    
        prev_bs, delex_context, context, user_uttr, in_domainslots = self.fert_decoder.encoder0(prev_bs, delex_context, context, user_uttr, in_domainslots)
        if not self.args['sep_dialog_history']:
            if 'delex_his' in self.args and self.args['delex_his']:
                out_slots = self.fert_decoder.domain2slot_nn(in_domainslots, None, context, b['context_mask'], delex_context, b['delex_context_mask']) 
            else:
                out_slots = self.fert_decoder.domain2slot_nn(in_domainslots, None, context, b['context_mask']) 
        else:
            if 'delex_his' in self.args and self.args['delex_his']:
                out_slots = self.fert_decoder.domain2slot_nn(in_domainslots, None, context, b['context_mask'], delex_context, b['delex_context_mask'], user_uttr, b['user_uttr_mask']) 
            else:
                out_slots = self.fert_decoder.domain2slot_nn(in_domainslots, None, context, b['context_mask'], user_uttr, b['user_uttr_mask']) 
        return out_slots 
    
    def make_input_tensor(self, tensor, factor, indices):
        if tensor is None: return tensor 
        return tensor.unsqueeze(1).expand(tensor.shape[0], factor, tensor.shape[1], tensor.shape[2]).reshape(-1, tensor.shape[1], tensor.shape[2])[indices]
    
    def generate_state_logits_atrg(self, b, out, y_in, delex_context2, context2):
        delex_context2, context2, y_in = self.state_decoder.encoder2(delex_context2, context2, y_in)
        context2 = self.make_input_tensor(context2, out['out_slots'].shape[1], b['sorted_in_domainslots2_idx'])
        context2_mask = self.make_input_tensor(b['context_mask'], out['out_slots'].shape[1], b['sorted_in_domainslots2_idx'])
        delex_context2 = self.make_input_tensor(delex_context2, out['out_slots'].shape[1], b['sorted_in_domainslots2_idx'])
        delex_context2_mask = self.make_input_tensor(b['delex_context_mask'], out['out_slots'].shape[1], b['sorted_in_domainslots2_idx'])
        
        out_states = self.state_decoder.slot2state_nn(y_in, b['y_mask'], context2, context2_mask,  delex_context2, delex_context2_mask) 
        return out_states, context2 
    
    
    def generate_state_logits(self, b, out_slots, prev_bs2, delex_context2, context2, user_uttr2, in_domainslots2): 
        prev_bs2, delex_context2, context2, user_uttr2, in_domainslots2 = self.state_decoder.encoder2(prev_bs2, delex_context2, context2, user_uttr2, in_domainslots2)

        if not self.args['sep_dialog_history']:
            if 'delex_his' in self.args and self.args['delex_his']:
                out_states = self.state_decoder.slot2state_nn(in_domainslots2, b['sorted_in_domainslots_mask'], context2, b['context_mask'],  delex_context2, b['delex_context_mask'])                       
            else:
                out_states = self.state_decoder.slot2state_nn(in_domainslots2, b['sorted_in_domainslots_mask'], context2, b['context_mask']) 
        else:
            if 'delex_his' in self.args and self.args['delex_his']:
                out_states = self.state_decoder.slot2state_nn(in_domainslots2, b['sorted_in_domainslots_mask'], context2, b['context_mask'], delex_context2, b['delex_context_mask'], user_uttr2, b['user_uttr_mask']) 
            else:
                out_states = self.state_decoder.slot2state_nn(in_domainslots2, b['sorted_in_domainslots_mask'], context2, b['context_mask'], user_uttr2, b['user_uttr_mask']) 
        return out_states

class Fertility_Decoder(nn.Module):
    def __init__(self, encoder0, encoder1,
             context_nn, domain2slot_nn, slot2lenval_nn,
             lenval_gen, gate_gen,
             in_embed, out_embed, out_domain_embed):
        super(Fertility_Decoder, self).__init__()
        self.encoder0 = encoder0
        self.domain2slot_nn = domain2slot_nn
        self.context_nn = context_nn
        self.lenval_gen = lenval_gen
        self.gate_gen = gate_gen
        self.in_embed = in_embed
        self.out_embed = out_embed
        self.out_domain_embed = out_domain_embed
    
class State_Decoder(nn.Module):
    def __init__(self, encoder2,
                 slot2state_nn,
                 in_embed2, out_embed2, out_domain2_embed,
                 state_gen):
        super(State_Decoder, self).__init__()
        self.encoder2 = encoder2
        self.slot2state_nn = slot2state_nn
        self.in_embed2 = in_embed2
        self.out_embed2 = out_embed2
        self.out_domain2_embed = out_domain2_embed
        self.state_gen = state_gen

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab, bias=False, W=None):
        super(Generator, self).__init__()
        self.shared_weight = False
        self.emb_proj = None
        if W is not None:
            self.proj = W
            self.shared_weight = True
            if W.shape[1] != d_model:
                emb_proj = [nn.Linear(d_model, W.shape[1]), nn.ReLU()]
                self.emb_proj = nn.Sequential(*emb_proj)                           
        else:
            self.proj = nn.Linear(d_model, vocab)
        self.bias = bias 
        if bias:
            self.bias_proj = nn.Linear(d_model, vocab)
        #self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.shared_weight:
            if hasattr(self, 'emb_proj') and self.emb_proj is not None: 
                x = self.emb_proj(x)
            proj_x = x.matmul(self.proj.transpose(1,0))
        else:
            proj_x = self.proj(x)
        if self.bias:
            c = x.sum(1)
            bias = self.bias_proj(c)
            bias = bias.unsqueeze(1).repeat(1, x.shape[1],1)
            return proj_x + bias 
        return proj_x
        #return F.log_softmax(self.proj(x), dim=-1)
        
class PointerGenerator(nn.Module):
    def __init__(self, vocab_gen, pointer_attn, args):
        super(PointerGenerator, self).__init__()
        self.vocab_gen = vocab_gen
        if args['pointer_decoder']:
            self.pointer_gen_W = nn.Linear(args['d_model']*3, 1) 
            if pointer_attn is not None: 
                self.pointer_attn = pointer_attn 
        self.args = args

    def forward(self, ft, context, context_mask):
        #if type(self.vocab_gen) == Generator:
        vocab_attn = self.vocab_gen(ft['out_states'])
        #else:
        #    vocab_attn = ft['out_states'].matmul(self.vocab_gen.transpose(1,0)) 
        if not self.args['pointer_decoder']:
            #p_out = F.log_softmax(vocab_attn, dim=-1)
            return vocab_attn  
        encoded_context = ft['encoded_context2']
        encoded_in_domainslots = ft['encoded_in_domainslots2']
        if self.args['pointer_attn']:
            if 'no_param_pointer_attn' in self.args and self.args['no_param_pointer_attn']:
                scores = torch.matmul(ft['out_states'], encoded_context.transpose(2,1))
                pointer_attn = F.softmax(scores, dim=-1)
            else:
                self.pointer_attn(ft['out_states'], encoded_context, encoded_context, context_mask)

                if 'h_ptr_attn' not in self.args or self.args['h_ptr_attn'] == 1:
                    pointer_attn = self.pointer_attn.attn.squeeze(1)
                else:
                    pointer_attn, _ = self.pointer_attn.attn.max(dim=1)
                    pointer_attn = F.softmax(pointer_attn, dim=-1)
        else:
            pointer_attn = ft['pointer_attn']  
        p_vocab = F.softmax(vocab_attn, dim = -1)
        context_index = context.unsqueeze(1).expand_as(pointer_attn)
        p_context_ptr = torch.zeros(p_vocab.size()).cuda()
        p_context_ptr.scatter_add_(2, context_index, pointer_attn)
        expanded_pointer_attn = pointer_attn.unsqueeze(-1).repeat(1, 1, 1, encoded_context.shape[-1])
        context_vec = (encoded_context.unsqueeze(1).expand_as(expanded_pointer_attn) * expanded_pointer_attn).sum(2)
        p_gen_vec = torch.cat([ft['out_states'], context_vec, encoded_in_domainslots], -1)
        vocab_pointer_switches = nn.Sigmoid()(self.pointer_gen_W(p_gen_vec)).expand_as(p_context_ptr)
        p_out = (1 - vocab_pointer_switches) * p_context_ptr + vocab_pointer_switches * p_vocab
        return torch.log(p_out)
        
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class AttentionNet(nn.Module):
    def __init__(self, layer, N):
        super(AttentionNet, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        #self.mode = mode 

    def forward(self, in_txt0, in_mask0, in_txt1=None, in_mask1=None, in_txt2=None, in_mask2=None, in_txt3=None, in_mask3=None):
        out = None 
        for layer in self.layers:
            if out is not None: in_txt0 = out 
            out = layer(in_txt0, in_mask0, in_txt1, in_mask1, in_txt2, in_mask2, in_txt3, in_mask3)
        return self.norm(out)

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

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, gated_conn=False):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.gated_conn = gated_conn
        if gated_conn:
            self.g_w1 = nn.Linear(size, size)
            self.g_w2 = nn.Linear(size, size)
            self.g_w3 = nn.Linear(size, 1)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        out = sublayer(self.norm(x))
        if self.gated_conn:
            g_s = torch.sigmoid(self.g_w3(self.dropout(torch.tanh(self.g_w1(out) + self.g_w2(x)))))
            return g_s * x + (1 - g_s) * self.dropout(out)
        else:
            return x + self.dropout(out)

    def expand_forward(self, x, sublayer):
        out = self.dropout(sublayer(self.norm(x)))
        out = out.mean(1).unsqueeze(1).expand_as(x)
        return x + out 

    def nosum_forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x)))

class SubLayer(nn.Module):
    def __init__(self, size, attn, ff, dropout, gated_conn, nb_attn, no_self_attn=False):
        super(SubLayer, self).__init__()
        self.attn = attn
        self.ff = ff
        self.size = size 
        self.attn = clones(attn, nb_attn)
        self.no_self_attn = no_self_attn
        self.sublayer = clones(SublayerConnection(size, dropout, gated_conn), nb_attn+1)

    def forward(self, tgt, tgt_mask, in_txt1=None, in_mask1=None, in_txt2=None, in_mask2=None, in_txt3=None, in_mask3=None, in_txt4=None, in_mask4=None, in_txt5=None, in_mask5=None):
        out = self.sublayer[0](tgt, lambda tgt: self.attn[0](tgt, tgt, tgt, tgt_mask))
        if in_txt1 is not None:
            out = self.sublayer[1](out, lambda out: self.attn[1](out, in_txt1, in_txt1, in_mask1))
            if in_txt2 is not None:
                out = self.sublayer[2](out, lambda out: self.attn[2](out, in_txt2, in_txt2, in_mask2))
                if in_txt3 is not None:
                    out = self.sublayer[3](out, lambda out: self.attn[3](out, in_txt3, in_txt3, in_mask3))
                    if in_txt4 is not None:
                        out = self.sublayer[4](out, lambda out: self.attn[4](out, in_txt4, in_txt4, in_mask4))
                        if in_txt5 is not None:
                            out = self.sublayer[5](out, lambda out: self.attn[5](out, in_txt5, in_txt5, in_mask5))
                            return self.sublayer[6](out, self.ff)
                        else:
                            return self.sublayer[5](out, self.ff)
                    else:
                        return self.sublayer[4](out, self.ff)
                else:
                    return self.sublayer[3](out, self.ff)
            else:
                return self.sublayer[2](out, self.ff)
        else:
            return self.sublayer[1](out, self.ff)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores =torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_in=-1, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        if d_in < 0: 
            d_in = d_model
        self.linears = clones(nn.Linear(d_in, d_model), 3)
        self.linears.append(nn.Linear(d_model, d_in))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, d_out=-1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        if d_out < 0:
            d_out = d_model 
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        #self.register_buffer('lut', lut)
        self.d_model = d_model
        #self.register_buffer('d_model', d_model)

    def forward(self, x):
        #pdb.set_trace()
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

def get_predictions_atrg(data, generated_states, lang, domain_lang, slot_lang, predictions,
                   dontcare_out=[], 
                   predicted_delex_context='', gt_delex_context='',
                   generated_lenval=[]):
    domain_lang = domain_lang if domain_lang is not None else lang
    slot_lang = slot_lang if slot_lang is not None else lang
    in_domainslots2_idx = data['sorted_in_domainslots2_idx']
    for idx, dial_id in enumerate(data['ID']):
        dial_id = data['ID'][idx]
        turn_id = data['turn_id'][idx]
        belief_state = data['turn_belief'][idx]
        domains = data['sorted_in_domains'][idx]
        slots = data['sorted_in_slots'][idx]
        state_gathered = {}
        for l_idx in range(data['sorted_in_domains'].shape[1]):
            temp = idx*data['sorted_in_domains'].shape[1]+l_idx
            if temp not in in_domainslots2_idx: continue 
            state_idx = (in_domainslots2_idx == temp).nonzero().item()
            domain = domain_lang.index2word[domains[l_idx].item()]
            slot = slot_lang.index2word[slots[l_idx].item()]
            domain = domain.replace("_DOMAIN", "")
            slot = slot.replace("_SLOT", "")
            key = '{}-{}'.format(domain,slot)
            state = generated_states[state_idx]
            dec_state = ''
            for i in state:
                state_token = lang.index2word[i.item()]
                if 'PAD' in [state_token]: continue
                if 'SOS' in [state_token]: continue   
                if 'UNK' in [state_token]: continue   
                if 'dontcare' in [state_token]: continue
                if 'none' in [state_token]: continue     
                if 'EOS' in [state_token]: break
                dec_state += state_token + ' ' 
            if dec_state.strip()!= '': state_gathered[key] = dec_state
        if len(dontcare_out)>0:
            for out in dontcare_out:
                d, s = out
                domain = domain_lang.index2word[d]
                slot = slot_lang.index2word[s]
                domain = domain.replace("_DOMAIN", "")
                slot = slot.replace("_SLOT", "")
                key = '{}-{}'.format(domain,slot)
                state_gathered[key] = 'dontcare'
                
        predicted_state = []
        for k,v in state_gathered.items():
            predicted_state.append('{}-{}'.format(k, v.strip()))
        if dial_id not in predictions: predictions[dial_id] = {}
        if turn_id not in predictions[dial_id]: predictions[dial_id][turn_id] = {}
            
        label_state = []
        for s in belief_state:
            v = s.split('-')[-1]
            if v != 'none': 
                label_state.append(s)
                
        pred_lenval = []
        if len(generated_lenval)>0:
            pred_lenval = ' '.join([str(i.item()) for i in generated_lenval[idx]])
        
        item = {}
        item['context_plain'] = data['context_plain'][idx]
        item['delex_context'] = gt_delex_context
        item['predicted_delex_context'] = predicted_delex_context
        item['lenval'] = ' '.join([str(i.item()) for i in data['sorted_lenval'][idx]])
        item['predicted_lenval'] = pred_lenval
        item['turn_belief'] = sorted(label_state)
        item['predicted_belief'] = predicted_state      
        predictions[dial_id][turn_id] = item 
    
    return predictions
    
def get_predictions(data, 
                    in_domains, in_slots, 
                    generated_states, 
                    lang, domain_lang, slot_lang, predictions, 
                    dontcare_out=[], 
                   predicted_delex_context='', gt_delex_context='',
                   generated_lenval=[]):
    domain_lang = domain_lang if domain_lang is not None else lang
    slot_lang = slot_lang if slot_lang is not None else lang
    sorted_index = np.argsort(data['turn_id'])
    for idx in sorted_index:
    #for idx, dial_id in enumerate(data['ID']):
        dial_id = data['ID'][idx]
        turn_id = data['turn_id'][idx]
        belief_state = data['turn_belief'][idx]
        state = [lang.index2word[i.item()] for i in generated_states[idx]]
        domains = [domain_lang.index2word[i.item()] for i in in_domains[idx]]
        slots = [slot_lang.index2word[i.item()] for i in in_slots[idx]] 
        state_gathered = {}
        for d_idx, domain in enumerate(domains):
            slot = slots[d_idx]
            state_token = state[d_idx]
            if 'PAD' in [domain, slot, state_token]: continue
            if 'EOS' in [domain, slot, state_token]: continue   
            if 'SOS' in [domain, slot, state_token]: continue   
            if 'UNK' in [domain, slot, state_token]: continue   
            if 'dontcare' in [state_token]: continue
            if 'none' in [state_token]: continue 
            domain = domain.replace("_DOMAIN", "")
            slot = slot.replace("_SLOT", "")
            key = '{}-{}'.format(domain,slot)
            if key not in state_gathered: state_gathered[key] = ''
            if len(state_gathered[key].split())>0 and state_gathered[key].split()[-1] == state_token: continue 
            state_gathered[key] += state_token + ' '
        if len(dontcare_out)>0:
            for out in dontcare_out:
                d, s = out
                domain = domain_lang.index2word[d]
                slot = slot_lang.index2word[s]
                domain = domain.replace("_DOMAIN", "")
                slot = slot.replace("_SLOT", "")
                key = '{}-{}'.format(domain,slot)
                state_gathered[key] = 'dontcare'
        predicted_state = []
        for k,v in state_gathered.items():
            predicted_state.append('{}-{}'.format(k, v.strip()))
        if dial_id not in predictions: predictions[dial_id] = {}
        if turn_id not in predictions[dial_id]: predictions[dial_id][turn_id] = {}
        label_state = []
        for s in belief_state:
            v = s.split('-')[-1]
            if v != 'none': 
                label_state.append(s)
        pred_lenval = []
        if len(generated_lenval)>0:
            pred_lenval = ' '.join([str(i.item()) for i in generated_lenval[idx]])
            
        item = {}
        item['context_plain'] = data['context_plain'][idx]
        item['delex_context'] = gt_delex_context
        item['predicted_delex_context'] = predicted_delex_context
        item['lenval'] = ' '.join([str(i.item()) for i in data['sorted_lenval'][idx]])
        item['predicted_lenval'] = pred_lenval
        item['turn_belief'] = sorted(label_state)
        item['predicted_belief'] = predicted_state      
        predictions[dial_id][turn_id] = item

    return predictions

def get_input_from_generated_gates(in1, in2, out1):
    gen_in1 = []
    gen_in2 = []
    dontcare_out = []
    for idx, o in enumerate(out1):
        i1 = in1[idx].item()
        i2 = in2[idx].item()
        if o == GATES['dontcare']:
            dontcare_out.append((i2, i1))
            continue
        if o not in [GATES['dontcare'], GATES['none']]:
            gen_in1.append(i1)
            gen_in2.append(i2)
    return torch.Tensor(gen_in1).long().cuda(), torch.Tensor(gen_in2).long().cuda(), dontcare_out

def get_input_from_generated(seq1, seq2, seq3, seq4):
    out_dict = {}
    dontcare_out = []
    for i in range(seq1.shape[0]):
        freq = seq2[i].item()
        val = seq1[i].item()
        if seq3 is not None: 
            added_val = seq3[i].item()
        if seq4 is not None: 
            gate = seq4[i]
            if gate == GATES['none']:
                continue 
            if gate == GATES['dontcare']:
                dontcare_out.append((added_val, val))
                continue 
        if val in [0,1,2,3]: continue # slot values as EOS, SOS, UNK, or PAD
        if freq == 0: continue # frequency as zero 
        if seq3 is not None: 
            if (added_val, val) not in out_dict: out_dict[(added_val, val)] = 0
            out_dict[(added_val, val)] = max([freq, out_dict[(added_val, val)]])
        else:
            if val not in out_dict: out_dict[val] = 0
            out_dict[val] = max([freq, out_dict[val]])
    out = []
    added_out = []
    for val, freq in out_dict.items():
        for j in range(freq):
            if seq3 is not None:
                added_out.append(val[0])
                out.append(val[1])
            else:
                out.append(val)
    out = torch.Tensor(out).long().cuda()
    if seq3 is not None:
        added_out = torch.Tensor(added_out).long().cuda()
        return added_out, out, dontcare_out
    return out, dontcare_out

def get_intermediate_predictions(in_domains, in_slots, gates, lenvals, domain_lang, slot_lang, gt_gates, gt_lenvals):
    prediction = []
    y = []
    for idx, d in enumerate(in_domains):
        s = in_slots[idx].item()
        d = d.item()
        slot = slot_lang.index2word[s].replace("_SLOT", "")
        domain = domain_lang.index2word[d].replace("_DOMAIN", "")
        key = '{}-{}'.format(domain,slot)
        if gates is not None:
            gate = REVERSE_GATES[gates[idx].item()]
            gt_gate = REVERSE_GATES[gt_gates[idx].item()]
            pred = '{}-{}-{}'.format(key, gate, lenvals[idx])
            gt = '{}-{}-{}'.format(key, gt_gate, gt_lenvals[idx])
        else:
            pred = '{}-{}'.format(key, lenvals[idx])
            gt = '{}-{}'.format(key, gt_lenvals[idx])
        prediction.append(pred)
        y.append(gt)
    return prediction, y

def get_prev_bs_seq(turn_id, dial_id, predictions, in_lang, slot_list):
    if turn_id == 0: 
        return torch.Tensor(0).long().cuda()
    pred = predictions[dial_id][turn_id-1]['predicted_belief']
    temp = [None] * len(slot_list)
    for p in pred:
        d,s,v = p.split('-')
        key = '{}-{}'.format(d,s)
        index = slot_list.index(key)
        temp[index] = p
    out = []
    for t in temp:
        if t is None: continue 
        d,s,v = t.split('-')
        d = '{}_DOMAIN'.format(d)
        s = '{}_SLOT'.format(s)
        out.append(in_lang.word2index[d])
        out.append(in_lang.word2index[s])
        for vt in v.split():
            out.append(in_lang.word2index[vt])
    return torch.Tensor(out).long().cuda()

def get_delex_from_tagged_uttr(turn_data, tagged_uttrs, in_lang):
    turn_id = turn_data['turn_id'][0]
    ID = turn_data['ID'][0]
    delex_context = ''
    for i in range(turn_id+1):
        delex_context += tagged_uttrs[ID][str(i)]['predicted_delex_turn_uttr']
    delex_context_mask = None 
    delex_context = [in_lang.word2index[word] if word in in_lang.word2index else UNK_token for word in delex_context.split()]
    delex_context = torch.Tensor(delex_context).unsqueeze(0).long().cuda()
    return delex_context, delex_context_mask 

def get_delex_from_prediction(turn_data, predictions, in_lang, args):
    turn_id = turn_data['turn_id'][0]
    ID = turn_data['ID'][0]
    gt_delex_context = turn_data['delex_context_plain']
    if ID not in predictions or turn_id-1 not in predictions[ID]:
        return  turn_data['delex_context'].unsqueeze(0), turn_data['delex_context_mask'].unsqueeze(0), gt_delex_context, gt_delex_context
    prev_bs = predictions[ID][turn_id-1]['predicted_belief']
    context = turn_data['context_plain'][0].split()
    delex_context = copy.copy(context)
    gt_delex_context = gt_delex_context.split()
    assert len(context) == len(delex_context) == len(gt_delex_context)
    sys_sos_index = [idx for idx,t in enumerate(delex_context) if t == 'SOS'][1::2]
    user_sos_index = [idx for idx,t in enumerate(delex_context) if t == 'SOS'][::2]
    
    
    for bs in prev_bs: 
        bs_tokens = bs.split('-')
        d, s, v = bs_tokens[0], bs_tokens[1], '-'.join(bs_tokens[2:])
        ds = '-'.join([d,s])
        if v in ['yes', 'no']:
            if ds == 'hotel-internet': 
                v = 'internet wifi'
            elif ds == 'hotel-parking':
                v = 'parking'
            else:
                print(ds, v)            
        v_tokens = v.split()
    
        if args['gt_sys_uttr']:
            temp = user_sos_index[:-1]
            for idx, u_idx in enumerate(temp): # delexicalize user utterance except last one 
                s_idx = sys_sos_index[idx]
                for t_idx, token in enumerate(delex_context[u_idx:s_idx]):
                    pos = t_idx + u_idx
                    if len(delex_context[pos].split('-')) == 2: continue 
                    if token in v_tokens:
                        delex_context[pos] = ds

            temp = user_sos_index[1:]
            for idx, u_idx in enumerate(temp): # delexicalize system utterance
                s_idx = sys_sos_index[idx]
                for t_idx, token in enumerate(delex_context[s_idx:u_idx]):
                    pos = t_idx + s_idx
                    delex_context[pos] = gt_delex_context[pos]
        else:
            for t_idx, token in enumerate(delex_context[:user_sos_index[-1]]): # delexicalize all 
                if len(delex_context[t_idx].split('-')) == 2: continue 
                if token in v_tokens:
                    delex_context[t_idx] = ds
        
    for idx, token in enumerate(delex_context[user_sos_index[-1]:]): # get the original last user uttr
        pos = idx + user_sos_index[-1]
        delex_context[pos] = context[pos]
    out = []
    for token in delex_context:
        token_index = in_lang.word2index[token] if token in in_lang.word2index else in_lang.word2index['UNK']
        out.append(token_index)
    out = torch.Tensor(out).unsqueeze(0).long().cuda()
    return out, None, ' '.join(delex_context), ' '.join(gt_delex_context)

def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    
def predict(
    logits,
    data,
    model,
    lang, 
    domain_lang,
    slot_lang,
    predictions,
    oracle,
    in_lang,
    args,
    slot_list=None, 
    test_dial_id=None, test_turn_id=-1,
    latency=None, src_lens=None, tgt_lens=None):
    p = args['p_test']  # simulate probability of using the non-ground truth delex context
    ft_p = args['p_test_fertility'] # simulate probability of using the non-ground truth fertility
    if not model.args['sep_input_embedding']:
        domain_lang = in_lang
        slot_lang = in_lang 
    if not oracle:
        y_maxlen = model.fert_decoder.lenval_gen.proj.out_features
        c = copy.deepcopy
        sorted_index = np.argsort(data['turn_id'])
        #sorted_index = range(len(data['turn_id']))
        for i in sorted_index:
            start = time.time()
            turn_data = {}
            for k,v in data.items():
                turn_data[k] = c(v[i]) if v is not None else v
            if test_dial_id is not None and turn_data['ID'] != test_dial_id: continue 
            turn_data['turn_id'] = [turn_data['turn_id']]
            turn_data['ID'] = [turn_data['ID']]
            turn_data['turn_belief'] = [turn_data['turn_belief']]
            turn_data['context'] = turn_data['context'].unsqueeze(0)
            turn_data['context_mask'] = turn_data['context_mask'].unsqueeze(0)  
            turn_data['sorted_in_domains'] = turn_data['sorted_in_domains'].unsqueeze(0)
            turn_data['sorted_in_slots'] = turn_data['sorted_in_slots'].unsqueeze(0) 
            turn_data['sorted_lenval'] = turn_data['sorted_lenval'].unsqueeze(0) 
            turn_data['context_plain'] = [turn_data['context_plain']]
            turn_data['sorted_in_domains2'] = turn_data['sorted_in_domains2'].unsqueeze(0)
            turn_data['sorted_in_slots2'] = turn_data['sorted_in_slots2'].unsqueeze(0)
            turn_data['sorted_in_domainslots_mask'] = turn_data['sorted_in_domainslots_mask'].unsqueeze(0)
            if model.args['sep_dialog_history']:
                turn_data['user_uttr'] = turn_data['user_uttr'].unsqueeze(0) 
                turn_data['user_uttr_mask'] = turn_data['user_uttr_mask'].unsqueeze(0)  
            predicted_delex_context = ''
            gt_delex_context = ''
            
            if model.args['delex_his']:
                if np.random.uniform() < p:
                    delex_context, delex_context_mask, predicted_delex_context, gt_delex_context = get_delex_from_prediction(turn_data, predictions, in_lang, args)
                    turn_data['delex_context'] = delex_context
                    turn_data['delex_context_mask'] = delex_context_mask
                else: # use ground truth input 
                    turn_data['delex_context'] = turn_data['delex_context'].unsqueeze(0)
                    turn_data['delex_context_mask'] = turn_data['delex_context_mask'].unsqueeze(0) 
            
            out = model.decode1(turn_data)
            generated_gates = None 
            dontcare_out = []
            generated_lenval = []
            if np.random.uniform() < ft_p:    
                if model.args['slot_gating']:
                    generated_gates = model.fert_decoder.gate_gen(out['out_slots']).max(dim=-1)[1]
                    generated_gates = generated_gates.squeeze(0)
                if model.args['slot_lenval']:
                    generated_lenval = model.fert_decoder.lenval_gen(out['out_slots']).max(dim=-1)[1] 
                else:
                    generated_lenval = torch.tensor([1]*generated_gates.shape[0]).unsqueeze(0)
                generated_in_domains2, generated_in_slots2, dontcare_out = get_input_from_generated(
                    turn_data['sorted_in_slots'].squeeze(0), 
                    generated_lenval.squeeze(0), 
                    turn_data['sorted_in_domains'].squeeze(0), 
                    generated_gates) 

                if len(generated_in_domains2)==0: 
                    dial_id = turn_data['ID'][0]
                    turn_id = turn_data['turn_id'][0]
                    if dial_id not in predictions: predictions[dial_id] = {}
                    if turn_id not in predictions[dial_id]: predictions[dial_id][turn_id] = {}
                    label_state = []
                    for s in turn_data['turn_belief'][0]:
                        v = s.split('-')[-1]
                        if v != 'none': 
                            label_state.append(s)   
                    predictions[dial_id][turn_id]['turn_belief'] = label_state
                    predictions[dial_id][turn_id]['predicted_belief'] = []
                    continue 

                turn_data['sorted_in_domains2'] = generated_in_domains2.unsqueeze(0)
                turn_data['sorted_in_slots2'] = generated_in_slots2.unsqueeze(0)
                turn_data['sorted_in_domainslots_mask'] = None 
            
            if model.args['auto_regressive']:
                if model.args['slot_lenval']:
                    temp = [i.item() for i in generated_lenval.reshape(-1)]
                    in_domainslots2_idx = [idx for idx,i in enumerate(temp) if i!=0]
                else:
                    temp = [i.item() for i in generated_gates.reshape(-1)]
                    in_domainslots2_idx = [idx for idx,i in enumerate(temp) if i==0]
                #in_domainslots2_idx = [idx for idx,i in enumerate(temp)]
                turn_data['sorted_in_domainslots2_idx'] = torch.tensor(in_domainslots2_idx).cuda()
                context = model.make_input_tensor(turn_data['context'].unsqueeze(-1), turn_data['sorted_in_domains'].shape[1], turn_data['sorted_in_domainslots2_idx']).squeeze(-1)
                context_mask = model.make_input_tensor(turn_data['context_mask'], turn_data['sorted_in_domains'].shape[1], turn_data['sorted_in_domainslots2_idx'])
                
                y_in = torch.tensor([lang.word2index['SOS']]*len(in_domainslots2_idx)).unsqueeze(1).cuda()
                y_mask = make_std_mask(y_in, lang.word2index['PAD'])    
                
                for i in range(y_maxlen+1):
                    turn_data['y_in'] = y_in
                    turn_data['y_mask'] = y_mask
                    out = model.decode2(turn_data, out) 
                    _, generated_states = model.state_decoder.state_gen(out, context, context_mask)[:,-1].max(dim=-1) 
                    y_in = torch.cat([y_in, generated_states.unsqueeze(1)], dim=-1)
                    y_mask = make_std_mask(y_in, lang.word2index['PAD'])
                predictions = get_predictions_atrg(turn_data, y_in, lang, domain_lang, slot_lang, predictions,
                        dontcare_out=dontcare_out,                           
                        predicted_delex_context=predicted_delex_context, gt_delex_context=gt_delex_context,
                        generated_lenval=generated_lenval) 
            else:
                out = model.decode2(turn_data, out) 
                out_attn = None
                if test_dial_id is not None and test_turn_id!=-1:
                    if turn_data['ID'][0] == test_dial_id and turn_data['turn_id'][0] == test_turn_id: 
                        return turn_data 
                generated_states = model.state_decoder.state_gen(out, turn_data['context'], turn_data['context_mask']).max(dim=-1)[1]
                predictions = get_predictions(turn_data, 
                                              turn_data['sorted_in_domains2'], turn_data['sorted_in_slots2'],
                                              generated_states, 
                                              lang, domain_lang, slot_lang, 
                                              predictions, 
                                              dontcare_out, 
                                             predicted_delex_context, gt_delex_context,
                                             generated_lenval) 
            end = time.time()
            elapsed_time = end - start 
            src_len = (turn_data['context']!=1).sum().item()
            tgt_len = generated_in_domains2.shape[0]
            if latency is not None and src_lens is not None and tgt_lens is not None:
                latency.append(elapsed_time)
                src_lens.append(src_len)
                tgt_lens.append(tgt_len)
        return predictions, latency, src_lens, tgt_lens
    else:
        joint_lenval_acc, joint_gate_acc =0, 0
        _, generated_lenval = model.fert_decoder.lenval_gen(logits['out_slots']).max(dim=-1)
        lenval_compared = (generated_lenval == data['sorted_lenval'])
        lenval_matches = lenval_compared.sum().item()
        joint_lenval_acc = ((lenval_compared!=1).sum(-1)==0).sum().item()

        if model.args['slot_gating']:
            _, generated_gates = model.fert_decoder.gate_gen(logits['out_slots']).max(dim=-1)
            gate_compared = (generated_gates == data['sorted_gates'])
            gate_maches = gate_compared.sum().item()
            joint_gate_acc = ((gate_compared!=1).sum(-1)==0).sum().item()

        if model.args['auto_regressive']:
            context = model.make_input_tensor(data['context'].unsqueeze(-1), data['sorted_in_domains'].shape[1], data['sorted_in_domainslots2_idx']).squeeze()
            context_mask = model.make_input_tensor(data['context_mask'], data['sorted_in_domains'].shape[1], data['sorted_in_domainslots2_idx'])
            _, generated_states = model.state_decoder.state_gen(logits, context, context_mask).max(dim=-1) 
            state_compared = (generated_states == data['y_out'])
            state_matches = state_compared.sum().item()
            predictions = get_predictions_atrg(data, generated_states, lang, domain_lang, slot_lang, predictions)
        else:         
            _, generated_states = model.state_decoder.state_gen(logits, data['context'], data['context_mask']).max(dim=-1) 
            state_compared = (generated_states == data['sorted_generate_y'])
            state_matches = state_compared.sum().item()
            predictions = get_predictions(data, data['sorted_in_domains2'], data['sorted_in_slots2'], generated_states, lang, domain_lang, slot_lang, predictions, generated_lenval=generated_lenval)
        matches = {}
        matches['joint_lenval'] = joint_lenval_acc
        matches['joint_gate'] = joint_gate_acc
        return matches, predictions


class Evaluator:
    "Optim wrapper that implements rate."
    def __init__(self, slots):
        self.slots = slots

    def evaluate_metrics(self, all_prediction, split, turn=-1):
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                if turn>-1 and t!=turn: continue 
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv['predicted_belief']):
                    joint_acc += 1
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv['predicted_belief']), self.slots[split])
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv['predicted_belief']))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total!=0 else 0
        turn_acc_score = turn_acc / float(total) if total!=0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count!=0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0 
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0 
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC 

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1 
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            if len(pred)==0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count

def make_model(
        src_lang, tgt_lang,
        domain_lang, slot_lang, tag_lang, 
        len_val, len_slot_val,
        args):
    
    domain2slot_nn_N = args['domain2slot_nn_N']
    slot2state_nn_N = args['slot2state_nn_N']
    context_nn_N = args['context_nn_N']
    d_model = args['d_model']
    h = args['h_attn']
    dropout = args['drop']
    c = copy.deepcopy
    d_ff = d_model*4
    src_vocab = src_lang.n_words
    tgt_vocab = tgt_lang.n_words
    slot_vocab = slot_lang.n_words
    domain_vocab = domain_lang.n_words
   
    att = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
    position = PositionalEncoding(d_model, dropout=dropout)
    if 'd_embed' in args and args['d_embed']!=-1:
        in_embed = [Embeddings(args['d_embed'], src_vocab), nn.Linear(args['d_embed'], d_model), nn.ReLU(), c(position)]
    else:
        in_embed = [Embeddings(d_model, src_vocab), c(position)]
    in_embed = nn.Sequential(*in_embed)

    out_domain_embed =  None
    out_embed = None
    out_domain2_embed =  None
    out_embed2 = None
    in_embed2 = None
    if args['sep_input_embedding']:
        if args['no_pe_ds_emb1']:
            out_domain_embed = [Embeddings(d_model, domain_vocab)]
            out_embed = [Embeddings(d_model, slot_vocab)]
        else:
            out_domain_embed = [Embeddings(d_model, domain_vocab), c(position)]
            out_embed = [Embeddings(d_model, slot_vocab), c(position)]
        out_domain_embed = nn.Sequential(*out_domain_embed)
        out_embed = nn.Sequential(*out_embed)
        if args['sep_embedding']:
            if args['no_pe_ds_emb2']:
                out_domain2_embed = Embeddings(d_model, domain_vocab)
                out_embed2 = Embeddings(d_model, domain_vocab)
            else:
                out_domain2_embed = [Embeddings(d_model, domain_vocab), c(position)]
                out_domain2_embed = nn.Sequential(*out_domain2_embed)
                out_embed2 = [Embeddings(d_model, slot_vocab), c(position)]
                out_embed2 = nn.Sequential(*out_embed2)
        elif not args['no_pe_ds_emb1'] and args['no_pe_ds_emb2']:
            out_domain2_embed = out_domain_embed[0]
            out_embed2 = out_embed[0]
        elif args['no_pe_ds_emb1'] and not args['no_pe_ds_emb2']:
            out_domain2_embed = [out_domain_embed[0], c(position)]
            out_domain2_embed = nn.Sequential(*out_domain2_embed)
            out_embed2 = [out_embed[0], c(position)]
            out_embed2 = nn.Sequential(*out_embed2)
    if args['sep_context_embedding']:
        in_embed2 = c(in_embed)
    encoder_nb_layers=2
    if args['sep_dialog_history']:
        encoder_nb_layers += 1
    #if args['previous_belief_state']:
    #    encoder_nb_layers += 1
    if args['delex_his']:
        encoder_nb_layers += 1
    encoder0 = Encoder(d_model, nb_layers=encoder_nb_layers)
    encoder1 = None
    encoder2 = None 
    domain2slot_nn = None
    lenval_gen = None
    slot2lenval_nn = None 
    gate_gen = None 
    context_nn = None 
    #if not args['one_network_only']:
    encoder2 = c(encoder0)
    #if args['seperate_gating']: 
    #    encoder1 = c(encoder0)
    if context_nn_N > 0: 
        context_layer = SubLayer(d_model, c(att), c(ff), dropout, False, nb_attn=1) 
        context_nn = AttentionNet(context_layer, context_nn_N)
    domain2slot_nb_attn=2
    if args['sep_dialog_history']:
        domain2slot_nb_attn+=1
    #if args['previous_belief_state']: 
    #    domain2slot_nb_attn+=1 
    if args['delex_his']:
        domain2slot_nb_attn+=1
    domain2slot_layer = SubLayer(d_model, c(att), c(ff), dropout, False, nb_attn=domain2slot_nb_attn) #, no_self_attn=True)
    domain2slot_nn = AttentionNet(domain2slot_layer, domain2slot_nn_N)
    lenval_gen = Generator(d_model, len_val+1, args['lenval_output_bias']) #lenval+1 to include 0-length 
    if args['slot_gating']:
        gate_gen = Generator(d_model, len(GATES), args['gate_output_bias'])
    #if args['seperate_gating']:  
    #    slot2lenval_nn = c(domain2slot_nn) 
    slot2state_nb_attn=2
    #if args['output2input']: #and not args['one_network_only']:
    #    slot2state_nb_attn+=1
    if args['sep_dialog_history']:
        slot2state_nb_attn+=1
    #if args['previous_belief_state']: 
    #    slot2state_nb_attn+=1    
    if args['delex_his']: # and not args['no_delex_attn']:
        slot2state_nb_attn+=1
    slot2state_layer = SubLayer(d_model, c(att), c(ff), dropout, False, nb_attn=slot2state_nb_attn)
    slot2state_nn = AttentionNet(slot2state_layer, slot2state_nn_N) 
    if args['pointer_decoder']:
        pointer_attn = MultiHeadedAttention(args['h_ptr_attn'], d_model, dropout=args['pointer_attn_dropout'])
        if args['sep_output_embedding']:
            state_gen = PointerGenerator(Generator(d_model, src_vocab, args['state_output_bias']), pointer_attn, args)
        else:
            state_gen = PointerGenerator(Generator(d_model, src_vocab, args['state_output_bias'], in_embed[0].lut.weight), pointer_attn, args)
    else:
        state_gen = PointerGenerator(Generator(d_model, tgt_vocab, args['state_output_bias']), None, args)    
    tag_gen, tag_nn = None, None 
    tag_gen2 = None 

    model = NBT(
        encoder0=encoder0,
        encoder1=encoder1,
        encoder2=encoder2,
        in_embed=in_embed,
        in_embed2=in_embed2,
        out_embed=out_embed,
        out_embed2=out_embed2,
        out_domain_embed=out_domain_embed,
        out_domain2_embed=out_domain2_embed,
        context_nn=context_nn,
        domain2slot_nn=domain2slot_nn,
        slot2lenval_nn=slot2lenval_nn,
        lenval_gen=lenval_gen,
        gate_gen = gate_gen, 
        slot2state_nn=slot2state_nn,
        state_gen = state_gen,
        tag_gen = tag_gen,
        tag_gen2 = tag_gen2,
        tag_nn = tag_nn,
        args = args
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
class LossCompute:
    "A simple loss compute and train function."
    def __init__(self, model, 
        len_val_criterion, state_gen_criterion, gate_gen_criterion, tag_gen_criterion, 
        opt, fert_decoder_opt, state_decoder_opt, args):
        self.model = model
        self.len_val_criterion = len_val_criterion
        self.tag_gen2_criterion = nn.CrossEntropyLoss()
        self.state_gen_criterion = state_gen_criterion
        self.gate_gen_criterion = gate_gen_criterion
        self.tag_gen_criterion = tag_gen_criterion
        self.opt = opt 
        self.fert_decoder_opt = fert_decoder_opt
        self.state_decoder_opt = state_decoder_opt
        self.args = args

    def __call__(self, out, data, is_eval=False):
        loss = 0
        lenval_out_loss = torch.Tensor([0])
        gate_out_loss = torch.Tensor([0])
        state_out_loss = torch.Tensor([0])
        state_out_nb_tokens, slot_out_nb_tokens, gate_out_nb_tokens = -1, -1, -1

        if self.args['slot_lenval']:
            slot_out_nb_tokens = data['sorted_lenval'].view(-1).size()[0]
            lenval_out = self.model.fert_decoder.lenval_gen(out['out_slots'])
            lenval_out_loss = self.len_val_criterion(lenval_out.view(-1, lenval_out.size(-1)),
                        data['sorted_lenval'].view(-1))
            loss += lenval_out_loss

        if self.args['slot_gating']:
            gate_out_nb_tokens = data['sorted_gates'].view(-1).size()[0]
            gate_out = self.model.fert_decoder.gate_gen(out['out_slots'])
            gate_out_loss = self.gate_gen_criterion(gate_out.view(-1, gate_out.size(-1)),
                     data['sorted_gates'].view(-1))
            loss += gate_out_loss 

        #if self.args['auto_regressive']:
        if self.args['auto_regressive']:
            context = self.model.make_input_tensor(data['context'].unsqueeze(-1), data['sorted_in_domains'].shape[1], data['sorted_in_domainslots2_idx']).squeeze()
            context_mask = self.model.make_input_tensor(data['context_mask'], data['sorted_in_domains'].shape[1], data['sorted_in_domainslots2_idx'])
            state_out_nb_tokens = data['y_out'].view(-1).size()[0]
            state_out = self.model.state_decoder.state_gen(out, context, context_mask) 
            state_out_loss = self.state_gen_criterion(state_out.view(-1, state_out.size(-1)), data['y_out'].view(-1))
        else:
            state_out_nb_tokens = data['sorted_generate_y'].view(-1).size()[0]
            state_out = self.model.state_decoder.state_gen(out, data['context'], data['context_mask']) 
            state_out_loss = self.state_gen_criterion(state_out.view(-1, state_out.size(-1)),data['sorted_generate_y'].view(-1))
        loss += state_out_loss
        
        if not is_eval:
            loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        elif self.fert_decoder_opt is not None and self.state_decoder_opt is not None: 
            self.fert_decoder_opt.step()
            self.fert_decoder_opt.optimizer.zero_grad()
            self.state_decoder_opt.step()
            self.state_decoder_opt.optimizer.zero_grad()

        losses = {}
        losses['lenval_loss'] = lenval_out_loss.item()
        losses['gate_loss'] = gate_out_loss.item()
        losses['state_loss'] = state_out_loss.item()

        nb_tokens = {}
        nb_tokens['slot'] = slot_out_nb_tokens 
        nb_tokens['state'] = state_out_nb_tokens
        nb_tokens['gate'] = gate_out_nb_tokens
        
        return losses, nb_tokens 
