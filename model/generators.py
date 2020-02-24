import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
from utils.config import * 
from model.modules import * 

class Generator(nn.Module):
    def __init__(self, d_model, vocab, W=None):
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

    def forward(self, x):
        if self.shared_weight:
            if hasattr(self, 'emb_proj') and self.emb_proj is not None:
                x = self.emb_proj(x)
            proj_x = x.matmul(self.proj.transpose(1,0))
        else:
            proj_x = self.proj(x)
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
        vocab_attn = self.vocab_gen(ft['out_states'])
        if not self.args['pointer_decoder']:
            return vocab_attn
        encoded_context = ft['encoded_context2']
        encoded_in_domainslots = ft['encoded_in_domainslots2']
        self.pointer_attn(ft['out_states'], encoded_context, encoded_context, context_mask)
        pointer_attn = self.pointer_attn.attn.squeeze(1)
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


