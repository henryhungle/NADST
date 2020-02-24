import torch
import torch.nn as nn
import torch.nn.functional as F

class Fertility_Decoder(nn.Module):
    def __init__(self, encoder,
             nn, lenval_gen, gate_gen,
             in_embed, domain_embed, slot_embed, args):
        super(Fertility_Decoder, self).__init__()
        self.encoder = encoder
        self.nn = nn
        self.lenval_gen = lenval_gen
        self.gate_gen = gate_gen
        self.in_embed = in_embed
        self.domain_embed = domain_embed
        self.slot_embed = slot_embed
        self.args = args

    def forward(self, b):
        out = {}
        in_domainslots, delex_context, context = self.get_embedded(b)
        out['encoded_context'] = context
        out['encoded_delex_context'] = delex_context
        out['encoded_in_domainslots'] = in_domainslots
        out_slots = self.generate_slot_logits(b, out)
        out['out_slots'] = out_slots
        return out

    def get_embedded(self, b):
        if self.args['delex_his']:
            delex_context = self.in_embed(b['delex_context'])
        else:
            delex_context = None
        context = self.in_embed(b['context'])
        if not self.args['sep_input_embedding']:
            in_domains = self.in_embed(b['sorted_in_domains'])
            in_slots = self.in_embed(b['sorted_in_slots'])
        else:
            in_domains = self.domain_embed(b['sorted_in_domains'])
            in_slots = self.slot_embed(b['sorted_in_slots'])
        in_domainslots = in_domains + in_slots
        return in_domainslots, delex_context, context

    def generate_slot_logits(self, b, encoded):
        delex_context, context, in_domainslots = self.encoder(encoded['encoded_delex_context'], encoded['encoded_context'], encoded['encoded_in_domainslots'])
        if self.args['delex_his']:
            out_slots = self.nn(in_domainslots, None, context, b['context_mask'], delex_context, b['delex_context_mask'])
        else:
            out_slots = self.nn(in_domainslots, None, context, b['context_mask'])
        return out_slots
