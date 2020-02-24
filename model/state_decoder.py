import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

class State_Decoder(nn.Module):
    def __init__(self, encoder,
                 nn, state_gen,
                 domain_embed, slot_embed, args,
                 fert_decoder):
        super(State_Decoder, self).__init__()
        self.encoder = encoder
        self.nn = nn
        self.domain_embed = domain_embed
        self.slot_embed = slot_embed
        self.state_gen = state_gen
        self.fert_decoder = fert_decoder
        self.args = args

    def forward(self, b, out):
        if self.args['auto_regressive']:
            y_in, delex_context2, context2 = self.get_embedded_atrg(b, out)
            out_states, context2 = self.generate_state_logits_atrg(b, out, y_in, delex_context2, context2)
            out['out_states'] = out_states
            out['encoded_context2'] = context2
            out['encoded_in_domainslots2'] = y_in
            return out
        else:
            in_domainslots2, delex_context2, context2 = self.get_embedded(b, out) 
            out_states = self.generate_state_logits(b, out, delex_context2, context2, in_domainslots2)
            out['out_states'] = out_states
            if self.args['pointer_decoder']:
                out['encoded_context2'] = context2
                out['encoded_in_domainslots2'] = in_domainslots2
        return out

    def get_embedded_atrg(self, b, encoded): 
        delex_context, context = encoded['encoded_delex_context'], encoded['encoded_context']
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

    def get_embedded(self, b, encoded): 
        delex_context2 = encoded['encoded_delex_context']
        context2 = encoded['encoded_context']

        if not self.args['sep_input_embedding']:
            in_domains2 = self.fert_decoder.in_embed(b['sorted_in_domains2'])
            in_slots2 = self.fert_decoder.in_embed(b['sorted_in_slots2'])
        else:
            if ((not self.args['no_pe_ds_emb1'] and self.args['no_pe_ds_emb2']) or \
                (self.args['no_pe_ds_emb1'] and not self.args['no_pe_ds_emb2'])):
                in_domains2 = self.domain_embed(b['sorted_in_domains2'])
                in_slots2 = self.slot_embed(b['sorted_in_slots2'])
            else:
                in_domains2 = self.fert_decoder.domain_embed(b['sorted_in_domains2'])
                in_slots2 = self.fert_decoder.slot_embed(b['sorted_in_slots2'])
        in_domainslots2 = in_domains2 + in_slots2
        return in_domainslots2, delex_context2, context2

    def make_input_tensor(self, tensor, factor, indices):
        if tensor is None: return tensor
        return tensor.unsqueeze(1).expand(tensor.shape[0], factor, tensor.shape[1], tensor.shape[2]).reshape(-1, tensor.shape[1], tensor.shape[2])[indices]

    def generate_state_logits_atrg(self, b, out, y_in, delex_context2, context2):
        delex_context2, context2, y_in = self.encoder(delex_context2, context2, y_in)
        context2 = self.make_input_tensor(context2, out['out_slots'].shape[1], b['sorted_in_domainslots2_idx'])
        context2_mask = self.make_input_tensor(b['context_mask'], out['out_slots'].shape[1], b['sorted_in_domainslots2_idx'])
        delex_context2 = self.make_input_tensor(delex_context2, out['out_slots'].shape[1], b['sorted_in_domainslots2_idx'])
        delex_context2_mask = self.make_input_tensor(b['delex_context_mask'], out['out_slots'].shape[1], b['sorted_in_domainslots2_idx'])

        out_states = self.nn(y_in, b['y_mask'], context2, context2_mask,  delex_context2, delex_context2_mask)
        return out_states, context2

    def generate_state_logits(self, b, out, delex_context2, context2, in_domainslots2):
        delex_context2, context2, in_domainslots2 = self.encoder(delex_context2, context2, in_domainslots2)

        if 'delex_his' in self.args and self.args['delex_his']:
            out_states = self.nn(in_domainslots2, b['sorted_in_domainslots_mask'], context2, b['context_mask'],  delex_context2, b['delex_context_mask'])
        else:
            out_states = self.nn(in_domainslots2, b['sorted_in_domainslots_mask'], context2, b['context_mask'])
        return out_states
