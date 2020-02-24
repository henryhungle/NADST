import numpy as np 
import math, copy, time 
import torch 

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
        temp = user_sos_index[:-1]
        for idx, u_idx in enumerate(temp): 
            s_idx = sys_sos_index[idx]
            for t_idx, token in enumerate(delex_context[u_idx:s_idx]):
                pos = t_idx + u_idx
                if len(delex_context[pos].split('-')) == 2: continue
                if token in v_tokens:
                    delex_context[pos] = ds
        temp = user_sos_index[1:]
        for idx, u_idx in enumerate(temp): 
            s_idx = sys_sos_index[idx]
            for t_idx, token in enumerate(delex_context[s_idx:u_idx]):
                pos = t_idx + s_idx
                delex_context[pos] = gt_delex_context[pos]

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

def predict(logits, data, model,
    lang, domain_lang, slot_lang,
    predictions, oracle,
    in_lang, args,
    slot_list=None, test_dial_id=None, test_turn_id=-1,
    latency=None, src_lens=None, tgt_lens=None):
    p = args['p_test']  # probability of using the non-ground truth delex context
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

            out = model.fert_decoder(turn_data)
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
                    out = model.state_decoder(turn_data, out)
                    _, generated_states = model.state_decoder.state_gen(out, context, context_mask)[:,-1].max(dim=-1)
                    y_in = torch.cat([y_in, generated_states.unsqueeze(1)], dim=-1)
                    y_mask = make_std_mask(y_in, lang.word2index['PAD'])
                predictions = get_predictions_atrg(turn_data, y_in, lang, domain_lang, slot_lang, predictions,
                        dontcare_out=dontcare_out,
                        predicted_delex_context=predicted_delex_context, gt_delex_context=gt_delex_context,
                        generated_lenval=generated_lenval)
            else:
                out = model.state_decoder(turn_data, out)
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
				

