import torch.nn.functional as F
from torch import nn


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
