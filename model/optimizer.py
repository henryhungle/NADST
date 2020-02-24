import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable

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
        len_val_criterion, state_gen_criterion, gate_gen_criterion, opt, args):
        self.model = model
        self.len_val_criterion = len_val_criterion
        self.tag_gen2_criterion = nn.CrossEntropyLoss()
        self.state_gen_criterion = state_gen_criterion
        self.gate_gen_criterion = gate_gen_criterion
        self.opt = opt
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
        
        losses = {}
        losses['lenval_loss'] = lenval_out_loss.item()
        losses['gate_loss'] = gate_out_loss.item()
        losses['state_loss'] = state_out_loss.item()

        nb_tokens = {}
        nb_tokens['slot'] = slot_out_nb_tokens
        nb_tokens['state'] = state_out_nb_tokens
        nb_tokens['gate'] = gate_out_nb_tokens

        return losses, nb_tokens
