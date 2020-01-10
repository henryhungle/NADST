import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
import ast
from collections import Counter
from collections import OrderedDict
from tqdm import tqdm
import os
import pickle
import pdb 
import numpy as np
import copy 
from random import shuffle
from torch.autograd import Variable
from .fix_label import *

EXPERIMENT_DOMAINS = None

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
      
    def index_words(self, sent, type):
        if type == 'utter':
            for word in sent.split():
                self.index_word(word)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
        elif type == 'domain':
            for domain in sent:
                self.index_word(domain)
        elif type == 'word':
            for w in sent:
                self.index_word(w)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
                for v in value.split():
                    self.index_word(v)
        elif type == 'domain_only':
            for slot in sent:
                d,s = slot.split('-')
                self.index_word('{}_DOMAIN'.format(d))
        elif type == 'slot_only':
            for slot in sent:
                d,s = slot.split('-')
                self.index_word('{}_SLOT'.format(s))
        elif type == 'domain_tag':
            for slot in sent:
                d,s = slot.split('-')
                self.index_word(d)
        elif type == 'slot_tag':
            for slot in sent:
                d,s = slot.split('-')
                self.index_word(s)
        elif type == 'domain_w2i':
            for k,v in sent.items():
                if v<4: continue 
                self.index_word('{}_DOMAIN'.format(k))
        elif type == 'slot_w2i':
            for k,v in sent.items():
                if v<4: continue 
                self.index_word('{}_SLOT'.format(k))

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_lang, mem_lang, domain_lang, slot_lang, tag_lang, args, split, ALL_SLOTS):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.delex_dialog_history = data_info['delex_dialog_history']
        self.user_uttr = data_info['user_uttr']
        self.turn_belief = data_info['turn_belief']
        self.turn_belief_dict = data_info['turn_belief_dict']
        self.sorted_domainslots = data_info['sorted_domainslots']
        self.turn_uttr = data_info['turn_uttr']
        self.sorted_in_domains = data_info['sorted_in_domains']
        self.sorted_in_slots = data_info['sorted_in_slots']
        self.sorted_in_domains2 = data_info['sorted_in_domains2']
        self.sorted_in_slots2 = data_info['sorted_in_slots2']
        if args['auto_regressive']:
            self.sorted_in_domainslots2_idx = data_info['sorted_in_domainslots2_idx']
            self.atrg_generate_y = data_info['atrg_generate_y']
        self.sorted_lenval = data_info['sorted_lenval']
        self.sorted_gates = data_info['sorted_gates']
        self.sorted_generate_y = data_info['sorted_generate_y']
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_lang.word2index
        self.mem_word2id = mem_lang.word2index
        self.domain_word2id = domain_lang.word2index
        self.slot_word2id = slot_lang.word2index
        self.all_slots = ALL_SLOTS
        self.args = args
        self.split = split 
        
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ID = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        turn_belief_dict = self.turn_belief_dict[index]
        sorted_domainslots = self.sorted_domainslots[index]
        turn_uttr = self.turn_uttr[index]
        context_plain = self.dialog_history[index] 
        sorted_lenval = self.sorted_lenval[index]
        sorted_in_domains2 = self.sorted_in_domains2[index]
        sorted_in_slots2 = self.sorted_in_slots2[index]
        sorted_generate_y = self.sorted_generate_y[index]
        c = copy.deepcopy
        context = self.preprocess(context_plain, self.src_word2id)
        delex_context = None
        if self.args['delex_his']:
            temp = self.delex_dialog_history[index].split()
            original = self.dialog_history[index].split()
            if self.split == 'train' and 'p_delex_noise' in self.args and np.random.uniform() < self.args['p_delex_noise']:
                prob = np.random.uniform()
                if prob < 0.5:
                    indices = [idx for idx,i in enumerate(temp) if len(i.split('-'))==2]
                    if len(indices) > 0:
                        random_idx = random.choice(indices)
                        temp[random_idx] = original[random_idx] # removal 
                else:
                    random_token = random.choice(self.all_slots)
                    out_words = list(self.mem_word2id.keys())[4:]
                    indices = [idx for idx,i in enumerate(original) if i in out_words]
                    if len(indices) > 0:
                        index = random.choice(indices)
                        temp[index] = random_token
            delex_context = ' '.join(temp)
            delex_context = self.preprocess(delex_context, self.src_word2id)   
        tag_x, tag_y = None, None
        if not self.args['sep_input_embedding']:
            sorted_in_domains = self.preprocess_seq(self.sorted_in_domains[index], self.src_word2id)
            sorted_in_slots = self.preprocess_seq(self.sorted_in_slots[index], self.src_word2id)
            sorted_in_domains2 = self.preprocess_seq(sorted_in_domains2, self.src_word2id)
            sorted_in_slots2 = self.preprocess_seq(sorted_in_slots2, self.src_word2id)
        else:
            sorted_in_domains = self.preprocess_seq(self.sorted_in_domains[index], self.domain_word2id)
            sorted_in_slots = self.preprocess_seq(self.sorted_in_slots[index], self.slot_word2id)
            sorted_in_domains2 = self.preprocess_seq(sorted_in_domains2, self.domain_word2id)
            sorted_in_slots2 = self.preprocess_seq(sorted_in_slots2, self.slot_word2id)
        sorted_in_domainslots2_idx, y_in, y_out = None, None, None
        if args['auto_regressive']:
            sorted_in_domainslots2_idx = self.sorted_in_domainslots2_idx[index]
            y_in, y_out = self.preprocess_atrg_seq(self.atrg_generate_y[index], self.src_word2id) 
        if self.args['pointer_decoder']:
            sorted_generate_y = self.preprocess_seq(sorted_generate_y, self.src_word2id)
        else:
            sorted_generate_y = self.preprocess_seq(sorted_generate_y, self.mem_word2id)
        sorted_gates = None
        if self.sorted_gates[index] is not None:
            sorted_gates = self.sorted_gates[index]
        user_uttr_plain, user_uttr = None, None
        turn_prev_bs_plain, turn_prev_bs = None, None
        
        item_info = {
            "ID":ID, 
            "turn_id":turn_id, 
            "turn_belief":turn_belief, #?
            "context":context,
            "delex_context_plain": self.delex_dialog_history[index],
            "delex_context": delex_context,
            "context_plain":context_plain, 
            "user_uttr": user_uttr,
            "user_uttr_plain": user_uttr_plain, 
            "sorted_in_domains": sorted_in_domains,
            "sorted_in_domains2": sorted_in_domains2,
            "sorted_in_slots": sorted_in_slots,
            "sorted_in_slots2": sorted_in_slots2,
            "sorted_in_domainslots2_idx": sorted_in_domainslots2_idx, 
            "sorted_lenval": sorted_lenval,
            "sorted_gates": sorted_gates,
            "sorted_generate_y": sorted_generate_y,
            "y_in": y_in,
            "y_out": y_out
            }
        return item_info

    def __len__(self):
        return self.num_total_seqs
    
    def augment_domains_slots(self, sorted_lenval, sorted_domainslots, turn_belief_dict,
                              domains2, slots2, y):
        if self.split == 'train' and self.args['p_data_augment_add_ds']>0: 
            p = np.random.uniform()
            if p <= self.args['p_data_augment_remove_ds']:
                num_rounds = np.random.randint(1, self.args['max_num_augment_remove_ds']+1)
                indices = np.random.choice(range(len(sorted_lenval)), num_rounds)
                for index in indices:
                    if sorted_lenval[index] > 0: 
                        sorted_lenval[index] = 0
                        domainslot = sorted_domainslots[index]
                        turn_belief_dict.pop(domainslot)
                        domains2, slots2, y = get_sorted_generate_y(sorted_domainslots, sorted_lenval, turn_belief_dict) 
            p = np.random.uniform()
            if p <= self.args['p_data_augment_add_ds']:
                num_rounds = np.random.randint(1, self.args['max_num_augment_add_ds']+1)
                indices = np.random.choice(range(len(sorted_lenval)), num_rounds)
                for index in indices:
                    if sorted_lenval[index] == 0:
                        lenval = np.random.randint(1, self.args['max_num_augment_add_ds_tokens']+1)
                        sorted_lenval[index] = lenval
                        domainslot = sorted_domainslots[index]
                        turn_belief_dict[domainslot] = ' '.join(['PAD']*lenval)
                        domains2, slots2, y = get_sorted_generate_y(sorted_domainslots, sorted_lenval, turn_belief_dict)    
        return domains2, slots2, y
                                          
    def augment_prev_bs(self, turn_prev_bs_plain):
        if self.split == 'train' and (self.args['p_data_augment_remove']>0 or self.args['p_data_augment_add']>0): 
            p = np.random.uniform()
            if p <= self.args['p_data_augment_remove']:
                num_rounds = np.random.randint(1, self.args['max_num_augment_remove_tokens']+1)
                num_rounds = min(num_rounds, len(turn_prev_bs_plain.split()))
                tokens = turn_prev_bs_plain.split()
                for aug_round in range(num_rounds):
                    remove_index = np.random.randint(0, len(tokens))
                    tokens = [t for idx, t in enumerate(tokens) if idx!=remove_index]
                turn_prev_bs_plain = ' '.join(tokens)
            p = np.random.uniform()
            if p <= self.args['p_data_augment_add']:
                num_rounds = np.random.randint(1, self.args['max_num_augment_add_tokens']+1)
                tokens = turn_prev_bs_plain.split()
                adding_tokens = np.random.choice(self.prev_bs_vocab, num_rounds)
                for aug_round in range(num_rounds):
                    add_index = np.random.randint(0, len(tokens)+1)
                    new_token = adding_tokens[aug_round]
                    tokens.insert(add_index, new_token)
                turn_prev_bs_plain = ' '.join(tokens)    
        return turn_prev_bs_plain
    
    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        story = torch.Tensor(story)
        return story

    def preprocess_tuple(self, sequence, word2idx, word2idx2=None):
        out1 = []
        out2 = []
        for s in sequence:
            s1, s2 = s
            out1.append(word2idx[s1] if s1 in word2idx else UNK_token)
            if type(s2) == str:
                if word2idx2:
                    out2.append(word2idx2[s2] if s2 in word2idx2 else UNK_token)
                else:
                    out2.append(word2idx[s2] if s2 in word2idx else UNK_token)
            else:
                out2.append(s2)
        out1 = torch.Tensor(out1)
        out2 = torch.Tensor(out2)
        return out1, out2

    def preprocess_seq(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            #v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + [EOS_token]
            story.append(word2idx[value] if value in word2idx else UNK_token)
        story = torch.Tensor(story)
        return story

    def preprocess_atrg_seq(self, seqs, word2idx):
        y_in = []
        y_out = []
        for seq in seqs:
            seq = ['SOS'] + seq + ['EOS']
            seq = self.preprocess_seq(seq, word2idx)
            y_in.append(seq[:-1])
            y_out.append(seq[1:])
        return y_in, y_out 
    
    def preprocess_memory(self, sequence, word2idx):
        """Converts words to ids."""
        stor = []
        for value in sequence:
            d, s, v = value
            s = s.replace("book","").strip()
            # separate each word in value to different memory slot
            for wi, vw in enumerate(v.split()):
                idx = [word2idx[word] if word in word2idx else UNK_token for word in [d, s, "t{}".format(wi), vw]]
                story.append(idx)
        story = torch.Tensor(story)
        return story
        
def collate_fn(data):
    def merge(sequences, pad_token, max_len=-1):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        if max_len < 0: 
            max_len = 1 if max(lengths)==0 else max(lengths)
        else:
            assert max_len >= max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long() * pad_token
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if type(seq) == list: 
                padded_seqs[i, :end] = torch.Tensor(seq[:end])
            else:
                padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
        return padded_seqs, lengths
    
    def merge_2d(seqs, pad_token):
        temp = []
        for seq in seqs:
            temp += seq
        return merge(temp, pad_token)
    
    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def merge_memory(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths) # avoid the empty belief state issue
        padded_seqs = torch.ones(len(sequences), max_len, 4).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if len(seq) != 0:
                padded_seqs[i,:end,:] = seq[:end]
        return padded_seqs, lengths 

    def get_mask(seqs, pad_token):
        return (seqs != pad_token).unsqueeze(-2)
    
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

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'], PAD_token)
    src_seqs_mask = get_mask(src_seqs, PAD_token)
    delex_context, delex_context_mask = None, None 
    if item_info['delex_context'][0] is not None:
        delex_context, _ = merge(item_info['delex_context'], PAD_token)
        delex_context_mask = get_mask(delex_context, PAD_token)
    user_uttr_seqs, user_uttr_mask = None, None
    if item_info['user_uttr'][0] is not None: 
        user_uttr_seqs, _ = merge(item_info['user_uttr'], PAD_token)
        user_uttr_mask = get_mask(user_uttr_seqs, PAD_token)   
    turn_prev_bs_seqs, turn_prev_bs_mask = None, None 
    src_sorted_domains_seqs, _ = merge(item_info['sorted_in_domains'], PAD_token)
    src_sorted_slots_seqs, _ = merge(item_info['sorted_in_slots'], PAD_token)
    src_sorted_domains2_seqs, _ = merge(item_info['sorted_in_domains2'], PAD_token)
    src_sorted_slots2_seqs, _ = merge(item_info['sorted_in_slots2'], PAD_token)
    src_sorted_domains2_seqs_mask = get_mask(src_sorted_domains2_seqs, PAD_token)                                              
    y_sorted_lenvals_seqs, _ = merge(item_info['sorted_lenval'], 0)
    y_sorted_seqs, _ = merge(item_info['sorted_generate_y'], PAD_token)
    y_in, y_out, y_mask, sorted_in_domainslots2_idx = None, None, None, None
    if item_info['y_in'][0] is not None:
        y_in, _ = merge_2d(item_info['y_in'], PAD_token)
        y_out, _ = merge_2d(item_info['y_out'], PAD_token)
        y_mask = make_std_mask(y_in, PAD_token)
        sorted_in_domainslots2_idx = []
        for idx, i in enumerate(item_info['sorted_in_domainslots2_idx']):
            temp = [ii + idx*src_sorted_domains_seqs.shape[1] for ii in i]
            sorted_in_domainslots2_idx += temp
        sorted_in_domainslots2_idx = torch.tensor(sorted_in_domainslots2_idx)

    y_sorted_gates_seqs = None
    if item_info['sorted_gates'][0] is not None: 
        y_sorted_gates_seqs, _ = merge(item_info['sorted_gates'], GATES['none'])
    
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        src_seqs_mask = src_seqs_mask.cuda()
        if item_info['user_uttr'][0] is not None: 
            user_uttr_seqs = user_uttr_seqs.cuda()
            user_uttr_mask = user_uttr_mask.cuda()
        src_sorted_domains_seqs = src_sorted_domains_seqs.cuda()
        src_sorted_slots_seqs = src_sorted_slots_seqs.cuda()
        src_sorted_domains2_seqs = src_sorted_domains2_seqs.cuda()
        src_sorted_slots2_seqs = src_sorted_slots2_seqs.cuda()
        src_sorted_domains2_seqs_mask = src_sorted_domains2_seqs_mask.cuda()
        y_sorted_lenvals_seqs = y_sorted_lenvals_seqs.cuda()
        y_sorted_seqs = y_sorted_seqs.cuda()
        if item_info['sorted_gates'][0] is not None:
            y_sorted_gates_seqs = y_sorted_gates_seqs.cuda()
        if item_info['delex_context'][0] is not None:    
            delex_context = delex_context.cuda()
            delex_context_mask = delex_context_mask.cuda()
        if item_info['y_in'][0] is not None:
            y_in = y_in.cuda()
            y_out = y_out.cuda()
            y_mask = y_mask.cuda()
            sorted_in_domainslots2_idx = sorted_in_domainslots2_idx.cuda()
        
    item_info["context"] = src_seqs
    item_info["context_mask"] = src_seqs_mask
    item_info["delex_context"] = delex_context
    item_info["delex_context_mask"] = delex_context_mask                                                 
    item_info['user_uttr'] = user_uttr_seqs
    item_info['user_uttr_mask'] = user_uttr_mask
    item_info['sorted_in_domains'] = src_sorted_domains_seqs
    item_info['sorted_in_slots'] = src_sorted_slots_seqs
    item_info['sorted_in_domains2'] = src_sorted_domains2_seqs
    item_info['sorted_in_slots2'] = src_sorted_slots2_seqs
    item_info['sorted_in_domainslots2_idx'] = sorted_in_domainslots2_idx
    item_info['sorted_in_domainslots_mask'] = src_sorted_domains2_seqs_mask
    item_info['sorted_lenval'] = y_sorted_lenvals_seqs
    item_info['sorted_generate_y'] = y_sorted_seqs
    item_info['sorted_gates'] = y_sorted_gates_seqs
    item_info['y_in'] = y_in
    item_info['y_out'] = y_out
    item_info['y_mask'] = y_mask
    return item_info

def process_turn_belief_dict(turn_belief_dict, turn_domain_flow, ordered_slot):
    domain_numslots_ls = []
    slot_lenval_ls = []
    slotval_ls = []
    in_domain_ls= []
    in_slot_ls = [] 
    in_domainslot_ls = []
    domain_numslots = {}
    domainslot_lenval = {} 
    # (domain, #slot) processing
    for k,v in turn_belief_dict.items():
        d, s = k.split('-') 
        if d not in domain_numslots: domain_numslots[d] = 0 
        domain_numslots[d] += 1 
        if d not in domainslot_lenval: domainslot_lenval[d] = []
        domainslot_lenval[d].append((s,len(v.split())))
    for d in turn_domain_flow: 
        if d in domain_numslots:
            domain_numslots_ls.append((d, domain_numslots[d]))
        else:
            domain_numslots_ls.append((d, 0)) # for cases which domain is found but not slot i.e. police domain 
    # (slot, len_slotVal) processing 
    if ordered_slot == 'alphabetical': 
        for k,v in domainslot_lenval.items():
            sorted_v = sorted(v, key=lambda tup: tup[0])
            domainslot_lenval[k] = sorted_v
    for dn in domain_numslots_ls:
        domain, numslots = dn
        for n in range(numslots):
            slot_lenval_ls.append((domainslot_lenval[domain][n]))
            in_domain_ls.append(domain)
    # (domain_slot, Val) processing 
    for i, d in enumerate(in_domain_ls):
        s, len_v = slot_lenval_ls[i]
        slotval_ls += turn_belief_dict["{}-{}".format(d,s)].split()
        for l in range(len_v):
            in_domainslot_ls.append((d,s))
    assert len(in_domain_ls) == len(slot_lenval_ls)
    assert len(in_domainslot_ls) == len(slotval_ls)
    return domain_numslots_ls, in_domain_ls, slot_lenval_ls, in_domainslot_ls, slotval_ls

def fix_book_slot_name(turn_belief_dict, slots):
    out = {}
    for k in turn_belief_dict.keys():
        new_k = k.replace(" ", "")
        if new_k not in slots: pdb.set_trace()
        out[new_k] = turn_belief_dict[k]
    return out 

def fix_multival(turn_belief_dict, multival_count):
    has_multival = False
    for k,v in turn_belief_dict.items():
        if '|' in v:
            values = v.split('|')
            turn_belief_dict[k] =  values[0] #' ; '.join(values)
            has_multival = True
    if has_multival: multival_count += 1 
    return turn_belief_dict, multival_count

def remove_none_value(turn_belief_dict):
    out = {}
    for k,v in turn_belief_dict.items():
        if v != 'none': 
            out[k] = v
    return out

def get_turn_domain_flow(turn_domain_flow, turn_belief_dict, ordered_domain):
    all_domains = set()
    for k in turn_belief_dict.keys():
        d, s = k.split('-')
        all_domains.add(d)
    new_domain = set(all_domains) - set(turn_domain_flow)
    #if len(new_domain) > 1: pdb.set_trace() # assume that each utt only have max. one new domain 
    if len(new_domain) == 0: return turn_domain_flow
    #turn_domain = list(new_domain)[0]
    for turn_domain in new_domain:
        if turn_domain not in turn_domain_flow: #assume each domain only appear once i.e. no case hotel -> restaurant -> hotel 
            turn_domain_flow += [turn_domain]
        elif turn_domain_flow[-1] != turn_domain: #move the latest domain to the last position 
            turn_domain_flow.remove(turn_domain)
            turn_domain_flow += [turn_domain]
    if ordered_domain == 'alphabetical':
        turn_domain_flow = sorted(turn_domain_flow)
    return turn_domain_flow

def get_sorted_lenval(sorted_domainslots, turn_belief_dict, slot_gating):
    sorted_lenval = [0] * len(sorted_domainslots)
    if slot_gating:
        sorted_gates = [GATES['none']] * len(sorted_domainslots)
    else:
        sorted_gates = None
    for k,v in turn_belief_dict.items():
        index = sorted_domainslots.index(k)
        lenval = len(v.split())
        if not slot_gating or (slot_gating and v not in ['dontcare', 'none']):
            sorted_lenval[index] = lenval
        if slot_gating:
            if v not in ['dontcare', 'none']:
                sorted_gates[index] = GATES['gen']
            else:
                sorted_gates[index] = GATES[v]
    return sorted_lenval, sorted_gates

def get_filtered_sorted_lenval(sorted_domainslots, sorted_gates, sorted_lenval):
    in_domains = []
    in_slots = []
    out_vals = []
    in_domainslots = []
    for idx, gate in enumerate(sorted_gates):
        if gate in [GATES['dontcare'], GATES['none']]: continue 
        lenval = sorted_lenval[idx]
        if lenval == 0: continue 
        domain = sorted_domainslots[idx].split('-')[0]
        slot = sorted_domainslots[idx].split('-')[1]
        in_domains.append(domain+"_DOMAIN")
        in_slots.append(slot+"_SLOT")
        in_domainslots.append(sorted_domainslots[idx])
        out_vals.append(lenval)
    return in_domains, in_slots, in_domainslots, out_vals
    
def get_sorted_generate_y(sorted_domainslots, sorted_lenval_ls, turn_belief_dict):
    in_domains = []
    in_slots = []
    in_domainslots_index = []
    out_vals = []
    for idx, lenval in enumerate(sorted_lenval_ls):
        if lenval == 0: continue 
        domain = sorted_domainslots[idx].split('-')[0]
        slot = sorted_domainslots[idx].split('-')[1]
        val = turn_belief_dict[sorted_domainslots[idx]].split()
        assert len(val) == lenval
        for i in range(lenval):
            in_domains.append(domain+"_DOMAIN")
            in_slots.append(slot+"_SLOT")
            out_vals.append(val[i])
            in_domainslots_index.append(idx)
    return in_domains, in_slots, out_vals, in_domainslots_index
                
def get_bs_seq(sorted_domainslots, turn_belief_dict):
    bs_seq = []
    for ds in sorted_domainslots:
        if ds in turn_belief_dict:
            d,s = ds.split('-')
            v = turn_belief_dict[ds]
            bs_seq.append('{}_DOMAIN'.format(d))
            bs_seq.append('{}_SLOT'.format(s))
            bs_seq += v.split()
    bs_seq = ' '.join(bs_seq)
    return bs_seq

def get_bs_noise_seq(sorted_domainslots, bs_noise):
    bs_noise_dict = {}
    for n in bs_noise:
        d,s,v = n.split('-')
        bs_noise_dict['{}-{}'.format(d,s)]=v
    bs_seq = get_bs_seq(sorted_domainslots, bs_noise_dict)
    return bs_seq

def get_act_dict(sys_act, domain):
    out = {}
    for act in sys_act: 
        if type(act)==str: continue 
        k, v = act 
        k = '-'.join([domain, k])
        if k in out:
            if type(out[k])==list:
                out[k].append(v)
            else:
                out[k] = [out[k], v]
        else:
            out[k] = v 
    return out 

def delex_from_dict(sent, d):
    for k,v in d.items():
        if v in ['yes', 'no']:
            if k == 'hotel-internet': 
                v = ['internet', 'wifi']
            elif k == 'hotel-parking':
                v = 'parking'
            else:
                continue
        if type(v) == list:
            for vi in v:
                placeholder = ' '.join(len(vi.split()) * [k])
                sent = sent.replace(' {} '.format(vi), ' {} '.format(placeholder))
        else:
            placeholder = ' '.join(len(v.split()) * [k])
            sent = sent.replace(' {} '.format(v), ' {} '.format(placeholder))
    return sent 
    
def delexicalize(dialog_history, turn_belief_dict, data=None, args=None):
    out = copy.copy(dialog_history)
    out = delex_from_dict(out, turn_belief_dict)
    if data is not None and args is not None and args['sys_act']:
        sys_act = data['system_acts']
        domain = data['domain']
        act_dict = get_act_dict(sys_act, domain)
        out = delex_from_dict(out, act_dict)
    return out
 
def get_tag_y(delex_dialog_history, turn_belief_dict, sep_slot_tag):
    if sep_slot_tag:
        out_domain = []
        out_slot = []
        for token in delex_dialog_history.split():
            if token in turn_belief_dict:
                d,s = token.split('-')
                out_domain.append(d)
                out_slot.append(s)
            else:
                out_domain.append('UNK')
                out_slot.append('UNK')
        return ' '.join(out_domain), ' '.join(out_slot)
    else:
        out = []
        for token in delex_dialog_history.split():
            if token in turn_belief_dict:
                out.append(token)
            else:
                out.append('UNK')
        out = ' '.join(out)
        return out 

def get_atrg_generate_y(sorted_domainslots, sorted_lenval_ls, turn_belief_dict):
    vals = []
    indices = []
    for idx, lenval in enumerate(sorted_lenval_ls):
        if lenval == 0: continue 
        val = turn_belief_dict[sorted_domainslots[idx]].split()
        assert len(val) == lenval
        vals.append(val)
        indices.append(idx)
    return vals, indices 
    
def read_langs(file_name, SLOTS, dataset, lang, mem_lang, training, args):
    print(("Reading from {}".format(file_name)))
    data = []
    max_resp_len, max_value_len = 0, 0
    max_nb_domains = 0
    max_nb_slots_per_domain = 0
    max_len_val_per_slot = 0
    max_len_slot_val = {}
    domain_counter = {} 
    count_noise = 0
    sorted_domainslots = sorted(SLOTS)
    sorted_in_domains = [i.split('-')[0]+"_DOMAIN" for i in sorted_domainslots]
    sorted_in_slots = [i.split('-')[1]+"_SLOT" for i in sorted_domainslots]
    for ds in sorted_domainslots:
        max_len_slot_val[ds] = (1, "none") # counting none/dontcare 
    
    multival_count = 0
    
    with open(file_name) as f:
        dials = json.load(f)
        # create vocab first 
        for dial_dict in dials:
            if (dataset=='train' and training) or (args['pointer_decoder']):
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    lang.index_words(turn["system_transcript"], 'utter')
                    lang.index_words(turn["transcript"], 'utter')
        # determine training data ratio, default is 100%
        if training and dataset=="train" and args["data_ratio"]!=100:
            random.Random(10).shuffle(dials)
            dials = dials[:int(len(dials)*0.01*args["data_ratio"])]
        
        #cnt_lin = 1
        for dial_dict in dials:
            last_belief_dict = {}
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if EXPERIMENT_DOMAINS and domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1
                
            # Unseen domain setting
            if args["only_domain"] != "" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
               (args["except_domain"] != "" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]): 
                continue

            # Reading data
            turn_domain_flow = []
            prev_bs = ''
            dialog_history = ''
            delex_dialog_history = ''
            len_his = 0
            running_turn_ls = []
            prev_turn_belief_dict = {}
            
            for ti, turn in enumerate(dial_dict["dialogue"]):
                if len_his >= args['max_len_his']:  
                    len_his -=1 
                    #start_idx = [m.start() for m in re.finditer('SOS', dialog_history)][2]
                    temp = dialog_history.split()
                    delex_temp = delex_dialog_history.split()
                    start_idx = [i for i,t in enumerate(temp) if t == 'SOS'][2]
                    dialog_history= ' '.join(temp[start_idx:])
                    delex_dialog_history= ' '.join(delex_temp[start_idx:])
                    running_turn_ls = running_turn_ls[-args['max_len_his']:]     
                turn_id = turn["turn_idx"]
                if ti == 0:
                    user_sent = ' SOS ' + turn["transcript"] + ' EOS '
                    sys_sent = '' 
                else:
                    #if args['sep_system_special_token']: 
                    #    sys_sent = ' SOSS '  + turn["system_transcript"] + ' EOSS '
                    #else:
                    sys_sent = ' SOS '  + turn["system_transcript"] + ' EOS '
                    user_sent = 'SOS ' + turn["transcript"] + ' EOS '
                    len_his += 1
                turn_uttr = sys_sent + user_sent 
                turn_uttr_strip = turn_uttr.strip()
                running_turn_ls.append(turn_uttr_strip)
                dialog_history += sys_sent 
                delex_dialog_history += sys_sent
                if args['sep_dialog_history']:
                    source_text = dialog_history.strip()
                    source_user_text = user_sent.strip()
                    dialog_history += user_sent 
                    delex_dialog_history += user_sent
                else:
                    dialog_history += user_sent 
                    delex_dialog_history += user_sent
                    source_user_text = None
                    source_text = dialog_history.strip()
                
                turns = [""] * (args['max_len_his']+1) #including the latest turns  
                if args['sep_dialog_turn']:
                    start_idx = len(running_turn_ls)
                    turns[-start_idx:] = running_turn_ls
                
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS)
                #turn_belief_dict = get_label_dict(turn["belief_state"], False)
                turn_belief_dict = fix_book_slot_name(turn_belief_dict, SLOTS)
                turn_belief_dict, multival_count = fix_multival(turn_belief_dict, multival_count)
                turn_belief_dict = remove_none_value(turn_belief_dict)
                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "":
                        if (dataset=='train' and training) or (args['pointer_decoder']):
                            temp = ' '.join(SLOTS)
                            lang.index_words(temp, 'utter')
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] not in k])

                sorted_lenval, sorted_gates = get_sorted_lenval(sorted_domainslots, turn_belief_dict, args['slot_gating'])
                sorted_in_domains1, sorted_in_slots1 = None, None 
                sorted_in_domains2, sorted_in_slots2, sorted_generate_y, sorted_in_domainslots2_index = get_sorted_generate_y(sorted_domainslots, sorted_lenval, turn_belief_dict)
                
                if args['auto_regressive']:
                    atrg_generate_y, sorted_in_domainslots2_index = get_atrg_generate_y(sorted_domainslots, sorted_lenval, turn_belief_dict)
                else:
                    atrg_generate_y = None 
                
                turn_prev_bs = None
                turn_prev_bs_noise = None
                
                if args['previous_belief_state']:
                    turn_prev_bs = prev_bs 
                    prev_bs = get_bs_seq(sorted_domainslots, turn_belief_dict)
                    if (dataset=='train' and training) or (args['pointer_decoder']):
                        lang.index_words(turn_prev_bs, 'utter') 
                    turn_prev_bs_noise = ''
                    if args['bs_noise'] and turn_id > 0 and dataset=="train": 
                        #if dial_dict["dialogue_idx"] in bs_noise and str(turn_id-1) in bs_noise[dial_dict["dialogue_idx"]]: 
                        noise = bs_noise[dial_dict["dialogue_idx"]][str(turn_id-1)]
                        if set(noise['turn_belief']) != set(noise['predicted_belief']):
                            turn_prev_bs_noise=get_bs_noise_seq(sorted_domainslots, noise['predicted_belief'])           
                            count_noise += 1
                #delex_dialog_history = None 
                
                tag_y = None 
                if args['delex_his']:
                    #if args['prev_turn_belief_dict']:
                    #    delex_dialog_history = delexicalize(delex_dialog_history, prev_turn_belief_dict, turn, args)
                    #else:
                    delex_dialog_history = delexicalize(delex_dialog_history, turn_belief_dict)
                    if args['partial_delex_his']:
                        temp = dialog_history.split()
                        delex_temp = delex_dialog_history.split()
                        start_idx = [i for i,t in enumerate(temp) if t == 'SOS'][-1] #delex all except the last user utterance
                        delex_dialog_history= ' '.join(delex_temp[:start_idx] + temp[start_idx:])
                    if len(delex_dialog_history.split()) != len(dialog_history.split()): pdb.set_trace()
                    if (dataset=='train' and training) or (args['pointer_decoder']):
                        lang.index_words(delex_dialog_history, 'utter')

                turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]
                for k,v in turn_belief_dict.items():
                    if len(v.split()) > max_len_slot_val[k][0]:
                        max_len_slot_val[k] = (len(v.split()), v)
                
                if dataset=='train' and training: 
                    mem_lang.index_words(turn_belief_dict, 'belief')

                data_detail = {
                    "ID":dial_dict["dialogue_idx"], 
                    "turn_id":turn_id, 
                    "dialog_history":source_text, 
                    "delex_dialog_history": delex_dialog_history,
                    "user_uttr": source_user_text,
                    "turn_belief":turn_belief_list,
                    "sorted_domainslots": sorted_domainslots,
                    "turn_belief_dict": turn_belief_dict, 
                    "turn_uttr":turn_uttr_strip, 
                    'sorted_in_domains': sorted_in_domains,
                    'sorted_in_slots': sorted_in_slots,
                    'sorted_in_domains2': sorted_in_domains2,
                    'sorted_in_slots2': sorted_in_slots2,
                    'sorted_in_domainslots2_idx': sorted_in_domainslots2_index, 
                    'sorted_lenval': sorted_lenval,
                    'sorted_gates': sorted_gates, 
                    'sorted_generate_y': sorted_generate_y,
                    'atrg_generate_y': atrg_generate_y
                    }
                data.append(data_detail)
                if len(sorted_lenval)>0 and max(sorted_lenval) > max_len_val_per_slot:
                    max_len_val_per_slot = max(sorted_lenval)
                prev_turn_belief_dict = turn_belief_dict

    print("domain_counter", domain_counter)
    print("multival_count", multival_count) 
    print("noise", count_noise)
    
    #print("max_slot_len_val", max_len_slot_val)
    return data, slot_temp, max_len_val_per_slot, max_len_slot_val


def get_seq(pairs, lang, mem_lang, domain_lang, slot_lang, tag_lang, batch_size, shuffle, args, split, ALL_SLOTS):  

    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k]) 

    dataset = Dataset(data_info, lang, mem_lang, domain_lang, slot_lang, tag_lang, args, split, ALL_SLOTS)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  collate_fn=collate_fn)
    return data_loader


def dump_pretrained_emb(word2index, index2word, dump_path):
    pdb.set_trace()
    print("Dumping pretrained embeddings...")
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)


def get_slot_information(ontology):
    if not EXPERIMENT_DOMAINS:
        ontology_domains = dict([(k, v) for k, v in ontology.items()])
    else:
        ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    #SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    slots = [k.replace(" ","").lower() for k in ontology_domains.keys()]
    all_domains = [i.split('-')[0] for i in slots]
    all_domains = set(all_domains)
    return slots, all_domains

def merge_lang(lang, max_freq):
    out = Lang()
    for k,v in lang.index2word.items():
        if k<4: 
            continue 
        else:
            for f in range(max_freq+1): #including length/size 0 
                out.index_word((v,f))
    return out

def get_onenet_langs(dials, max_len_slot_val, args):
    sorted_domainslots = dials[0]['sorted_domainslots']
    in_domains = []
    in_slots = []
    for k in sorted_domainslots: 
        d,s = k.split('-')
        max_len = max_len_slot_val[k]
        for l in range(max_len):
            in_domains.append(d + "_DOMAIN")
            in_slots.append(s + "_SLOT") 
    for idx, dial in enumerate(dials):
        dials[idx]["sorted_in_domains_all"] = None
        dials[idx]["sorted_in_slots_all"] = None
        dials[idx]["sorted_generate_y_all"] = None 
    return dials

def dump_pretrained_emb(word2index, index2word, dump_path):
    print("Dumping pretrained embeddings...")
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)
        
def prepare_data_seq(training, args):
    batch_size = args['batch']
    eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size
    file_train = 'data{}/train_dials.json'.format(args['data_version'])
    file_dev = 'data{}/dev_dials.json'.format(args['data_version'])
    file_test = 'data{}/test_dials.json'.format(args['data_version'])
    ontology = json.load(open("data2.0/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    ALL_SLOTS, ALL_DOMAINS = get_slot_information(ontology)
    lang, mem_lang = Lang(), Lang()
    domain_lang, slot_lang = None, None
    domain_lang = Lang()
    slot_lang = Lang()
    tag_lang = Lang()
    tag_lang.index_words(ALL_SLOTS, 'word')
    lang.index_words(ALL_SLOTS, 'slot')
    mem_lang.index_words(ALL_SLOTS, 'slot')   
    lang.index_word('dontcare')
    mem_lang.index_word('dontcare')
    domain_lang.index_words(ALL_SLOTS, 'domain_only')
    slot_lang.index_words(ALL_SLOTS, 'slot_only')
    if training:
        pair_train, slot_train, train_max_len_val, train_max_len_slot_val = read_langs(file_train, ALL_SLOTS, "train", lang, mem_lang, training, args)
        pair_dev, slot_dev, dev_max_len_val, dev_max_len_slot_val = read_langs(file_dev, ALL_SLOTS, "dev", lang, mem_lang, training, args)
        pair_test, slot_test, test_max_len_val, test_max_len_slot_val = read_langs(file_test, ALL_SLOTS, "test", lang, mem_lang, training, args)
        max_len_slot_val={}
        for k,v in train_max_len_slot_val.items():
            max_len_slot_val[k] = max([train_max_len_slot_val[k][0], dev_max_len_slot_val[k][0], test_max_len_slot_val[k][0]])
        max_len_val = max(train_max_len_val, dev_max_len_val, test_max_len_val)
        if not args['sep_input_embedding'] or args['previous_belief_state']:
            lang.index_words(domain_lang.word2index, 'domain_w2i')
            lang.index_words(slot_lang.word2index, 'slot_w2i')
        pair_train = get_onenet_langs(pair_train, max_len_slot_val, args)
        pair_dev = get_onenet_langs(pair_dev, max_len_slot_val, args)
        pair_test = get_onenet_langs(pair_test, max_len_slot_val, args)
        train = get_seq(pair_train, lang, mem_lang, domain_lang, slot_lang, tag_lang, batch_size, True, args, 'train', ALL_SLOTS)
        dev   = get_seq(pair_dev, lang, mem_lang, domain_lang, slot_lang, tag_lang, eval_batch, False, args, 'dev', ALL_SLOTS)
        test  = get_seq(pair_test, lang, mem_lang, domain_lang, slot_lang, tag_lang, eval_batch, False, args, 'test', ALL_SLOTS)

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))  
    print("Vocab_size: %s " % lang.n_words)
    print("Vocab_size Training %s" % lang.n_words )
    print("Vocab_size Belief %s" % mem_lang.n_words )
    print("Vocab_size Domain {}".format(domain_lang.n_words))
    print("Vocab_size Slot {}".format(slot_lang.n_words))
    print("Max. len of value per slot: train {} dev {} test {} all {}".format(train_max_len_val, dev_max_len_val, test_max_len_val, max_len_val))
    print("USE_CUDA={}".format(USE_CUDA))

    SLOTS_LIST = {}
    SLOTS_LIST['all'] = ALL_SLOTS
    SLOTS_LIST['train'] = slot_train
    SLOTS_LIST['dev'] = slot_dev
    SLOTS_LIST['test'] = slot_test
    print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST['dev']))))
    print(SLOTS_LIST['dev'])
    print("[Test Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST['test']))))
    print(SLOTS_LIST['test'])
    
    return train, dev, test, lang, mem_lang, domain_lang, slot_lang, tag_lang, SLOTS_LIST, max_len_val, max_len_slot_val

