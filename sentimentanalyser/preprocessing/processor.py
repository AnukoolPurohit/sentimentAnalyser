import spacy
import collections
from spacy.symbols import ORTH
from sentimentanalyser.utils.preprocessing import default_pre_rules, default_post_rules
from sentimentanalyser.utils.preprocessing import default_spec_tok
from sentimentanalyser.utils.data import compose, parallel
from sentimentanalyser.utils.processor import uniquefy

class TokenizerProcessor():
    def __init__(self, lang='en', chunksize=2000, pre_rules=default_pre_rules,
                    post_rules=default_post_rules, max_workers=4):
        self.max_workers, self.chunksize = max_workers, chunksize
        self.pre_rules, self.post_rules = pre_rules, post_rules
        self.tokenizer = spacy.blank(lang).tokenizer
        for spec_tok in default_spec_tok:
            self.tokenizer.add_special_case(spec_tok, [{ORTH: spec_tok}])
    
    def process(self, chunk):
        "Process chunks of text items"
        chunk = [compose(txt, self.pre_rules) for txt in chunk]
        # because of this line we can't use proc1 as the base for this function
        docs = [[d.text for d in doc] for doc in self.tokenizer.pipe(chunk)]
        docs = [compose(toks, self.post_rules) for toks in docs]
        return docs
    
    def proc1(self, item):
        "Process a single item"
        return self.process([item])[0]
    
    def deproc1(self, toks):
        "Deprocess a single item"
        return ' '.join(toks)
    
    def deprocess(self, docs):
        "Deprocess multiple items"
        return [self.deproc1(toks) for toks in docs]
    
    def __call__(self, items):
        chunks = [items[i:i+self.chunksize] for i in range(0, len(items), self.chunksize)]
        toks = parallel(self.process, chunks, max_workers=self.max_workers)
        return sum(toks, [])


class NuemericalizeProcessor():
    def __init__(self, vocab=None, vocab_size=60000, min_freq=2):
        self.vocab, self.vocab_size, self.min_freq = vocab, vocab_size, min_freq
    
    def __call__(self, tokenslist):

        if self.vocab is None:
            freq = collections.Counter(token for tokens in tokenslist for token in tokens)
            self.vocab = [token for token,count in freq.most_common(self.vocab_size) if count >=self.min_freq]
            for spec_tok in reversed(default_spec_tok):
                if spec_tok in self.vocab:
                    self.vocab.remove(spec_tok)
                self.vocab.insert(0, spec_tok)
                
        if getattr(self, 'otoi', None) is None:
            self.otoi = collections.defaultdict(int, {v:k for k,v in enumerate(self.vocab)})
        
        return self.process(tokenslist)
    
    def process(self, tokenslist):
        return [self.proc1(tokens) for tokens in tokenslist]
    
    def proc1(self, tokens):
        return [self.otoi[token] for token in tokens]
    
    def deprocess(self, idxslist):
        return [self.deproc1(idxs) for idxs in idxslist]
    
    def deproc1(self, idxs):
        return [self.vocab(idx) for idx in idxs]


class CategoryProcessor():
    def __init__(self):
        self.vocab = None

    def __call__(self, items):
        if self.vocab is None:
            self.vocab = uniquefy(items)
            self.otoi = {v:k for k,v in enumerate(self.vocab)}
        return self.process(items)
    
    def process(self, items):
        return [self.proc1(item) for item in items]
    
    def proc1(self, item):
        return self.otoi[item]
    
    def deprocess(self, idxs):
        return [self.deproc1(idx) for idx in idxs]
    
    def deproc1(self, idx):
        return self.vocab[idx]