import torch
import torch.utils.data as data

import copy

from utils.config import * 

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_lang, mem_lang, domain_lang, slot_lang, args, split, ALL_SLOTS):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.delex_dialog_history = data_info['delex_dialog_history']
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
            delex_context = ' '.join(temp)
            delex_context = self.preprocess(delex_context, self.src_word2id)
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
        if self.args['auto_regressive']:
            sorted_in_domainslots2_idx = self.sorted_in_domainslots2_idx[index]
            y_in, y_out = self.preprocess_atrg_seq(self.atrg_generate_y[index], self.src_word2id)
        if self.args['pointer_decoder']:
            sorted_generate_y = self.preprocess_seq(sorted_generate_y, self.src_word2id)
        else:
            sorted_generate_y = self.preprocess_seq(sorted_generate_y, self.mem_word2id)
        sorted_gates = None
        if self.sorted_gates[index] is not None:
            sorted_gates = self.sorted_gates[index]

        item_info = {
            "ID":ID,
            "turn_id":turn_id,
            "turn_belief":turn_belief,
            "context":context,
            "delex_context_plain": self.delex_dialog_history[index],
            "delex_context": delex_context,
            "context_plain":context_plain,
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

    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        story = torch.Tensor(story)
        return story

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

