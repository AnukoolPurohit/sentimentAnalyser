import torch
import torch.nn.functional as F
from torch import nn
from sentimentanalyser.models.attention import WordSentenceAttention
from sentimentanalyser.models.regularization import WeightDropout
from sentimentanalyser.utils.models import get_info, get_embedding_vectors


class SimpleLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_size=300,
                 hidden_size=512, output_size=2, dropout_rate=0.5,
                 pad_idx=1, num_layers=1):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=pad_idx)
        self.lstm = nn.LSTM(hidden_size, hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)
        return

    def forward(self, xb):
        seq_lens, mask = get_info(xb, self.pad_idx)

        embedded = self.embedding(xb)
        embedded = self.dropout(embedded)

        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, seq_lens,
                                                       batch_first=True)

        packed_out, (hidden_state, cell_state) = self.lstm(packed_emb)
        out = self.linear(hidden_state.squeeze(0))
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_sz, embed_sz=300,
                 hidden_sz=512, output_sz=2, dropout=0.5, pad_idx=1,
                 num_layers=2, bidirectional=True):
        super().__init__()

        self.pad_idx = pad_idx
        self.embededing = nn.Embedding(vocab_sz, embed_sz, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(embed_sz, hidden_sz, batch_first=True,
                           bidirectional=bidirectional,
                           dropout=0. if num_layers == 1 else dropout,
                           num_layers=num_layers)

        self.bidir = 2 if bidirectional else 1

        self.attn = WordSentenceAttention(hidden_sz * self.bidir)

        self.linear = nn.Linear(self.bidir * hidden_sz, output_sz)
        return

    def forward(self, xb):
        seq_lens, mask = get_info(xb, self.pad_idx)

        embeded = self.embededing(xb)
        packed_embd = nn.utils.rnn.pack_padded_sequence(embeded, seq_lens,
                                                        batch_first=True)
        packed_out, (hidden_st, cell_st) = self.rnn(packed_embd)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        sentence = self.attn(outputs)
        return self.linear(sentence)


class PTLSTMModel(nn.Module):
    def __init__(self, local_vocab, torchtext_vocab,
                 hidden_sz=256, output_sz=2, dropout=0.5,
                 pad_idx=1, num_layers=2, bidirectional=True):
        super().__init__()

        self.pad_idx = pad_idx
        self.bidir = 2 if bidirectional else 1
        embd_vecs = get_embedding_vectors(local_vocab, torchtext_vocab)
        self.embedding = nn.Embedding.from_pretrained(embd_vecs,
                                                      freeze=False, padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(embd_vecs.shape[1], hidden_sz,
                           batch_first=True,
                           dropout=0. if num_layers == 1 else dropout,
                           num_layers=num_layers,
                           bidirectional=bidirectional)

        self.attn = WordSentenceAttention(self.bidir * hidden_sz)
        self.linear = nn.Linear(self.bidir * hidden_sz, output_sz)
        return

    def forward(self, xb):
        seq_lens, mask = get_info(xb, self.pad_idx)

        embeded = self.dropout(self.embedding(xb))
        packed_emb = nn.utils.rnn.pack_padded_sequence(embeded, seq_lens,
                                                       batch_first=True)

        packed_out, (hidden_st, cell_st) = self.rnn(packed_emb)

        context, _ = nn.utils.rnn.pad_packed_sequence(packed_out,
                                                      batch_first=True)

        sentence = self.attn(context)
        return self.linear(sentence)


class GloveConcatModel(nn.Module):
    def __init__(self, local_vocab, torchtext_vocab,
                 hidden_sz=256, output_sz=2, dropout=0.5,
                 num_layers=2, pad_idx=1, bidirectional=True):
        super().__init__()

        self.bidir = 2 if bidirectional else 1
        self.pad_idx = pad_idx

        embd_vecs = get_embedding_vectors(local_vocab, torchtext_vocab)
        self.embedding = nn.Embedding.from_pretrained(embd_vecs,
                                                      freeze=False,
                                                      padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(embd_vecs.shape[1], hidden_sz,
                           batch_first=True,
                           dropout=0. if num_layers == 1 else dropout,
                           num_layers=num_layers,
                           bidirectional=bidirectional)

        self.linear = nn.Linear((2 + num_layers) * hidden_sz * self.bidir,
                                output_sz)
        return

    def forward(self, xb):
        seq_lens, mask = get_info(xb, self.pad_idx)

        embedded = self.dropout(self.embedding(xb))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, seq_lens,
                                                   batch_first=True)

        packed_out, (hidden_st, cell_st) = self.rnn(packed)

        lstm_out, lens = nn.utils.rnn.pad_packed_sequence(packed_out)

        avg_pool = F.adaptive_avg_pool1d(lstm_out.permute((1, 2, 0)), 1).squeeze()
        max_pool = F.adaptive_max_pool1d(lstm_out.permute((1, 2, 0)), 1).squeeze()

        hidden_st = hidden_st.view(hidden_st.shape[1], -1)

        combined = torch.cat([max_pool, avg_pool, hidden_st], dim=1)
        return self.linear(combined)


class AWDModel(nn.Module):
    def __init__(self, local_vocab, torchtext_vocab,
                 hidden_sz=256, output_sz=2, dropout=0.5,
                 pad_idx=1, num_layers=2, bidirectional=True):
        super().__init__()

        self.pad_idx = pad_idx

        self.bidir = 2 if bidirectional else 1

        embd_vecs = get_embedding_vectors(local_vocab, torchtext_vocab)

        self.embeddings = nn.Embedding.from_pretrained(embd_vecs,
                                                       freeze=False,
                                                       padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)

        self.rnn = WeightDropout(nn.LSTM(embd_vecs.shape[1],
                                         hidden_sz,
                                         batch_first=True,
                                         dropout=0. if num_layers == 1 else dropout,
                                         num_layers=num_layers,
                                         bidirectional=bidirectional))

        self.linear = nn.Linear((2 + num_layers) * self.bidir * hidden_sz, output_sz)
        return

    def forward(self, xb):
        seq_lens, mask = get_info(xb, self.pad_idx)

        embedded = self.dropout(self.embeddings(xb))

        packed = nn.utils.rnn.pack_padded_sequence(embedded, seq_lens,
                                                   batch_first=True)

        packed_out, (hidden_st, cell_st) = self.rnn(packed)

        lstm_out, lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        avg_pool = F.adaptive_avg_pool1d(lstm_out.transpose(1, 2), 1).squeeze()
        max_pool = F.adaptive_max_pool1d(lstm_out.transpose(1, 2), 1).squeeze()

        hidden_st = hidden_st.view(hidden_st.shape[1], -1)

        combined = torch.cat([max_pool, avg_pool, hidden_st], dim=1)
        return self.linear(combined)


class AttnAWDModel(nn.Module):
    def __init__(self, local_vocab, torchtext_vocab,
                 hidden_sz=256, output_sz=2, dropout=0.5,
                 pad_idx=1, num_layers=2, bidirectional=True):
        super().__init__()

        self.pad_idx = pad_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        bidir = 2 if bidirectional else 1
        embd_vecs = get_embedding_vectors(local_vocab, torchtext_vocab)

        self.embedding = nn.Embedding.from_pretrained(embd_vecs,
                                                      freeze=False,
                                                      padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)

        self.rnn = WeightDropout(nn.LSTM(embd_vecs.shape[1],
                                         hidden_sz,
                                         batch_first=True,
                                         dropout=0. if num_layers == 1 else dropout,
                                         num_layers=num_layers,
                                         bidirectional=bidirectional))

        self.WSattn = WordSentenceAttention(bidir * hidden_sz)
        self.linear = nn.Linear(bidir * hidden_sz, output_sz)
        return

    def forward(self, xb):
        seq_lens, mask = get_info(xb, self.pad_idx)
        embedded = self.embedding(xb)
        embedded = self.dropout(embedded)
        packed_i = nn.utils.rnn.pack_padded_sequence(embedded, seq_lens,
                                                     batch_first=True)

        packed_o, (hidden_st, cell_st) = self.rnn(packed_i)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_o, batch_first=True)

        sentence = self.WSattn(outputs)
        return self.linear(sentence)
