import json
import unicodedata
import string
import re
import random
import time
import math
import ast 

import torch
import torch.utils.data as data 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

from utils.config import *
from utils.fix_label import * 
from utils.lang import * 
from utils.dataset import * 

from collections import Counter
from collections import OrderedDict
from tqdm import tqdm
import os
import pickle
import pdb 
import numpy as np
import copy 
from random import shuffle

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

'''
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
'''

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

'''
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
'''

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
    max_len_val_per_slot = 0
    max_len_slot_val = {}
    domain_counter = {} 
    #count_noise = 0
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
        
        for dial_dict in dials:
            last_belief_dict = {}
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1
                
            # Reading data
            dialog_history = ''
            delex_dialog_history = ''
            prev_turn_belief_dict = {}
            
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_id = turn["turn_idx"]
                if ti == 0:
                    user_sent = ' SOS ' + turn["transcript"] + ' EOS '
                    sys_sent = '' 
                    dlx_user_sent = ' SOS ' + turn["delex_transcript"] + ' EOS '
                    dlx_sys_sent = ''
                else:
                    sys_sent = ' SOS '  + turn["system_transcript"] + ' EOS '
                    user_sent = 'SOS ' + turn["transcript"] + ' EOS '
                    dlx_sys_sent = ' SOS '  + turn["delex_system_transcript"] + ' EOS '
                    dlx_user_sent = 'SOS ' + turn["delex_transcript"] + ' EOS '
                turn_uttr = sys_sent + user_sent 
                dialog_history += sys_sent 
                delex_dialog_history += dlx_sys_sent
                dialog_history += user_sent 
                delex_dialog_history += dlx_user_sent

                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS)
                turn_belief_dict = fix_book_slot_name(turn_belief_dict, SLOTS)
                turn_belief_dict, multival_count = fix_multival(turn_belief_dict, multival_count)
                turn_belief_dict = remove_none_value(turn_belief_dict)

                sorted_lenval, sorted_gates = get_sorted_lenval(sorted_domainslots, turn_belief_dict, args['slot_gating'])
                sorted_in_domains2, sorted_in_slots2, sorted_generate_y, sorted_in_domainslots2_index = get_sorted_generate_y(sorted_domainslots, sorted_lenval, turn_belief_dict)
                
                if args['auto_regressive']:
                    atrg_generate_y, sorted_in_domainslots2_index = get_atrg_generate_y(sorted_domainslots, sorted_lenval, turn_belief_dict)
                else:
                    atrg_generate_y = None 

                if args['delex_his']:
                    temp = dialog_history.split()
                    delex_temp = delex_dialog_history.split()
                    start_idx = [i for i,t in enumerate(temp) if t == 'SOS'][-1] #delex all except the last user utterance
                    in_delex_dialog_history= ' '.join(delex_temp[:start_idx] + temp[start_idx:])
                    if len(in_delex_dialog_history.split()) != len(dialog_history.split()): pdb.set_trace()
                    if (dataset=='train' and training) or (args['pointer_decoder']):
                        lang.index_words(in_delex_dialog_history, 'utter')

                turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]
                for k,v in turn_belief_dict.items():
                    if len(v.split()) > max_len_slot_val[k][0]:
                        max_len_slot_val[k] = (len(v.split()), v)
                
                if dataset=='train' and training: 
                    mem_lang.index_words(turn_belief_dict, 'belief')

                data_detail = {
                    "ID":dial_dict["dialogue_idx"], 
                    "turn_id":turn_id, 
                    "dialog_history": dialog_history.strip(), 
                    "delex_dialog_history": in_delex_dialog_history.strip(),
                    "turn_belief":turn_belief_list,
                    "sorted_domainslots": sorted_domainslots,
                    "turn_belief_dict": turn_belief_dict, 
                    "turn_uttr":turn_uttr.strip(), 
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
    
    return data, SLOTS, max_len_val_per_slot, max_len_slot_val


def get_seq(pairs, lang, mem_lang, domain_lang, slot_lang, batch_size, shuffle, args, split, ALL_SLOTS):  

    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k]) 

    dataset = Dataset(data_info, lang, mem_lang, domain_lang, slot_lang, args, split, ALL_SLOTS)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  collate_fn=collate_fn)
    return data_loader

def get_slot_information(ontology):
    #if not EXPERIMENT_DOMAINS:
    ontology_domains = dict([(k, v) for k, v in ontology.items()])
    #else:
    #ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
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


def prepare_data_seq(training, args):
    batch_size = args['batch']
    eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size
    file_train = 'data{}/nadst_train_dials.json'.format(args['data_version'])
    file_dev = 'data{}/nadst_dev_dials.json'.format(args['data_version'])
    file_test = 'data{}/nadst_test_dials.json'.format(args['data_version'])
    ontology = json.load(open("data2.0/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    ALL_SLOTS, ALL_DOMAINS = get_slot_information(ontology)
    lang, mem_lang = Lang(), Lang()
    domain_lang, slot_lang = Lang(), Lang()
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
        
        max_len_val = max(train_max_len_val, dev_max_len_val, test_max_len_val)
        
        if not args['sep_input_embedding']:
            lang.index_words(domain_lang.word2index, 'domain_w2i')
            lang.index_words(slot_lang.word2index, 'slot_w2i')
        
        train = get_seq(pair_train, lang, mem_lang, domain_lang, slot_lang, batch_size, True, args, 'train', ALL_SLOTS)
        dev   = get_seq(pair_dev, lang, mem_lang, domain_lang, slot_lang, eval_batch, False, args, 'dev', ALL_SLOTS)
        test  = get_seq(pair_test, lang, mem_lang, domain_lang, slot_lang, eval_batch, False, args, 'test', ALL_SLOTS)

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
    
    return train, dev, test, lang, mem_lang, domain_lang, slot_lang, SLOTS_LIST, max_len_val

