import torch


def to_detach(h):
    """Detaches `h` from its history."""
    return h.detach() if type(h) == torch.Tensor else tuple(to_detach(v) for v in h)


def get_lens_and_masks(x, pad_id=1):
    mask = (x == pad_id)
    sequence_lengths = x.size(1) - (x == pad_id).sum(1)
    return sequence_lengths, mask


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


def dropout_mask(x, sz, p):
    return x.new(*sz).bernoulli_(1-p).div_(1-p)
