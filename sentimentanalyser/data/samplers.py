from torch.utils.data import Sampler
from torch import LongTensor, tensor
import torch


class SortSampler(Sampler):
    def __init__(self, data_source, key):
        self.data_source, self.key = data_source, key
    
    def __len__(self):
        return len(self.data_source)
    
    def __iter__(self):
        return iter(sorted(list(range(len(self.data_source))), key=self.key, reverse=True))


class SortishSampler(Sampler):
    def __init__(self, data_source, key, bs):
        self.data_source, self.key, self.bs = data_source, key, bs
    
    def __len__(self) -> int:
        return len(self.data_source)
    
    def __iter__(self):
        idxs = torch.randperm(len(self.data_source))
        megabatches = [idxs[i:i+self.bs*50] for i in range(0, len(idxs), self.bs*50)]
        sorted_idx = torch.cat([tensor(sorted(s, key=self.key, reverse=True)) for s in megabatches])
        batches = [sorted_idx[i:i+self.bs] for i in range(0, len(sorted_idx), self.bs)]
        max_idx = torch.argmax(tensor([self.key(ck[0]) for ck in batches]))  # find the chunk with the largest key,
        batches[0], batches[max_idx] = batches[max_idx], batches[0]            # then make sure it goes first.
        batch_idxs = torch.randperm(len(batches)-2)
        sorted_idx = torch.cat([batches[i+1] for i in batch_idxs]) if len(batches) > 1 else LongTensor([])
        sorted_idx = torch.cat([batches[0], sorted_idx, batches[-1]])
        return iter(sorted_idx)
