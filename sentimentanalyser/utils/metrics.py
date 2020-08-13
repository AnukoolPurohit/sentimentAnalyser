import torch


def accuracy(preds, y):
    preds   = torch.argmax(preds, dim=1)
    correct = (preds == y).float()
    acc     = correct.sum() / len(correct)
    return acc