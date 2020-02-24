from utils.config import *

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


