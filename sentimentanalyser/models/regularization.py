import torch.nn.functional as F
from torch import nn
from .utils import dropout_mask


class EmbeddingsWithDropout(nn.Module):
    def __init__(self, embeddings, embeddings_dropout):
        super().__init__()
        self.embeddings = embeddings
        self.embeddings_dropout = embeddings_dropout
        self.padding_idx = self.embeddings.padding_idx
        if self.padding_idx is None:
            self.padding_idx = -1

    def forward(self, words, scale=None):
        if self.training and self.embeddings_dropout != 0:
            vocab_length, embedding_size = self.embeddings.weight.size()
            mask = dropout_mask(self.embeddings.weight.data,
                                (vocab_length, 1),
                                self.embeddings_dropout)

            masked_embeddings = self.embeddings.weight * mask

        else:
            masked_embeddings = self.embeddings.weight

        return F.embedding(words, masked_embeddings, self.padding_idx,
                           self.embeddings.max_norm, self.embeddings.norm_type,
                           self.embeddings.scale_grad_by_freq, self.embeddings.sparse)


class RNNDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, inp):
        if not self.training or self.dropout == 0.:
            return inp
        bs, seq_len, vocab_size = inp.size()
        mask = dropout_mask(inp.data, (bs, 1, vocab_size), self.dropout)
        return inp * mask


class WeightDropout(nn.Module):
    def __init__(self, module, dropout=0.5):
        super().__init__()
        self.module, self.dropout = module, dropout
        
        self.layer_names = self.get_layer_names()
        for layer_name in self.layer_names:
            weight = getattr(self.module, layer_name)
            self.register_parameter(layer_name+'_raw', nn.Parameter(weight.data))
        return
    
    def _setweights(self):
        for layer_name in self.layer_names:
            raw_w = getattr(self, layer_name+'_raw')
            self.module._parameters[layer_name] = F.dropout(raw_w, self.dropout,
                                                            training=self.training)
        return
    
    def get_layer_names(self):
        names = [f"weight_hh_l{i}" for i in range(self.module.num_layers)]
        if self.module.bidirectional:
            names = names + [name+'_reverse' for name in names]
        return names
    
    def forward(self, *args):
        self._setweights()
        self.module.flatten_parameters()
        return self.module.forward(*args)
