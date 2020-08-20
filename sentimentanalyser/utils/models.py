import torch


def get_info(x, pad_id=1):
    mask = (x == pad_id)
    lenghts = x.size(1) - (x == pad_id).sum(1)
    return lenghts, mask


def get_embedding_vectors(local_vocab, torchtext_vocab):
    size, dims = torchtext_vocab.vectors.shape
    vector_values = []
    for tok in local_vocab:
        if tok in torchtext_vocab.stoi:
            vector_values.append(torchtext_vocab.vectors[
                                 torchtext_vocab.stoi[tok]].unsqueeze(0))
        else:
            vector_values.append(torch.zeros(1, dims))
    assert len(local_vocab) == len(vector_values)
    return torch.cat(vector_values, dim=0)