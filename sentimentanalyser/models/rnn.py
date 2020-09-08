import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from .attention import WordSentenceAttention
from .regularization import WeightDropout, RNNDropout, EmbeddingsWithDropout
from .utils import get_lens_and_masks, get_embedding_vectors, to_detach


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
        seq_lens, mask = get_lens_and_masks(xb, self.pad_idx)

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
        self.embedding = nn.Embedding(vocab_sz, embed_sz, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(embed_sz, hidden_sz, batch_first=True,
                           bidirectional=bidirectional,
                           dropout=0. if num_layers == 1 else dropout,
                           num_layers=num_layers)

        self.directions = 2 if bidirectional else 1

        self.attn = WordSentenceAttention(hidden_sz * self.directions)

        self.linear = nn.Linear(self.directions * hidden_sz, output_sz)
        return

    def forward(self, xb):
        seq_lens, mask = get_lens_and_masks(xb, self.pad_idx)

        embedded = self.embedding(xb)
        packed_embed = nn.utils.rnn.pack_padded_sequence(embedded, seq_lens,
                                                         batch_first=True)
        packed_out, (hidden_st, cell_st) = self.rnn(packed_embed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        sentence = self.attn(outputs)
        return self.linear(sentence)


class PTLSTMModel(nn.Module):
    def __init__(self, local_vocab, torchtext_vocab,
                 hidden_sz=256, output_sz=2, dropout=0.5,
                 pad_idx=1, num_layers=2, bidirectional=True):
        super().__init__()

        self.pad_idx = pad_idx
        self.directions = 2 if bidirectional else 1
        embedding_vectors = get_embedding_vectors(local_vocab, torchtext_vocab)
        self.embedding = nn.Embedding.from_pretrained(embedding_vectors,
                                                      freeze=False, padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(embedding_vectors.shape[1], hidden_sz,
                           batch_first=True,
                           dropout=0. if num_layers == 1 else dropout,
                           num_layers=num_layers,
                           bidirectional=bidirectional)

        self.attn = WordSentenceAttention(self.directions * hidden_sz)
        self.linear = nn.Linear(self.directions * hidden_sz, output_sz)
        return

    def forward(self, xb):
        seq_lens, mask = get_lens_and_masks(xb, self.pad_idx)

        embedded = self.dropout(self.embedding(xb))
        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, seq_lens,
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

        self.directions = 2 if bidirectional else 1
        self.pad_idx = pad_idx

        embedding_vectors = get_embedding_vectors(local_vocab, torchtext_vocab)
        self.embedding = nn.Embedding.from_pretrained(embedding_vectors,
                                                      freeze=False,
                                                      padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(embedding_vectors.shape[1], hidden_sz,
                           batch_first=True,
                           dropout=0. if num_layers == 1 else dropout,
                           num_layers=num_layers,
                           bidirectional=bidirectional)

        self.linear = nn.Linear((2 + num_layers) * hidden_sz * self.directions,
                                output_sz)
        return

    def forward(self, xb):
        seq_lens, mask = get_lens_and_masks(xb, self.pad_idx)

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

        self.directions = 2 if bidirectional else 1

        embedding_vectors = get_embedding_vectors(local_vocab, torchtext_vocab)

        self.embeddings = nn.Embedding.from_pretrained(embedding_vectors,
                                                       freeze=False,
                                                       padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)

        self.rnn = WeightDropout(nn.LSTM(embedding_vectors.shape[1],
                                         hidden_sz,
                                         batch_first=True,
                                         dropout=0. if num_layers == 1 else dropout,
                                         num_layers=num_layers,
                                         bidirectional=bidirectional))

        self.linear = nn.Linear((2 + num_layers) * self.directions * hidden_sz, output_sz)
        return

    def forward(self, xb):
        seq_lens, mask = get_lens_and_masks(xb, self.pad_idx)

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

        directions = 2 if bidirectional else 1
        embedding_vectors = get_embedding_vectors(local_vocab, torchtext_vocab)

        self.embedding = nn.Embedding.from_pretrained(embedding_vectors,
                                                      freeze=False,
                                                      padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)

        self.rnn = WeightDropout(nn.LSTM(embedding_vectors.shape[1],
                                         hidden_sz,
                                         batch_first=True,
                                         dropout=0. if num_layers == 1 else dropout,
                                         num_layers=num_layers,
                                         bidirectional=bidirectional))

        self.attention = WordSentenceAttention(directions * hidden_sz)
        self.linear = nn.Linear(directions * hidden_sz, output_sz)
        return

    def forward(self, xb):
        seq_lens, mask = get_lens_and_masks(xb, self.pad_idx)
        embedded = self.embedding(xb)
        embedded = self.dropout(embedded)
        packed_i = nn.utils.rnn.pack_padded_sequence(embedded, seq_lens,
                                                     batch_first=True)

        packed_o, (hidden_st, cell_st) = self.rnn(packed_i)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_o, batch_first=True)

        sentence = self.attention(outputs)
        return self.linear(sentence)


class AWDLSTM(nn.Module):

    def __init__(self, embedding_size, hidden_size, num_layers, weight_drop=0.5,
                 hidden_drop=0.2):
        super().__init__()
        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding_size, self.batch_size = embedding_size, 1
        self.rnns = []
        self.hidden_dropouts = []

        for layer in range(num_layers):
            input_size = embedding_size if layer == 0 else hidden_size
            output_size = hidden_size if layer != num_layers else embedding_size
            rnn = nn.LSTM(input_size, output_size, num_layers=1, batch_first=True)
            self.rnns.append(WeightDropout(rnn, weight_drop))
            self.hidden_dropouts.append(RNNDropout(hidden_drop))

        self.rnns = nn.ModuleList(self.rnns)
        self.hidden_dropouts = nn.ModuleList(self.hidden_dropouts)

    def _one_hidden(self, layer):
        """Return one hidden state."""
        nh = self.hidden_size if layer != self.num_layers - 1 else self.embedding_size
        return next(self.parameters()).new(1, self.batch_size, nh).zero_()

    def reset(self):
        """Reset the hidden states."""
        self.hidden = [(self._one_hidden(layer), self._one_hidden(layer))
                       for layer in range(self.num_layers)]

    def forward(self, embedded):
        batch_size, seq_len, vocab_size = embedded.size()
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.reset()

        new_hidden, raw_outputs, outputs = [], [], []

        raw_output = embedded
        for layer, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dropouts)):
            raw_output, new_h = rnn(raw_output, self.hidden[layer])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if layer != self.num_layers - 1:
                raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden)
        return raw_outputs, outputs


class LinearDecoder(nn.Module):
    def __init__(self, hidden_sz, output_sz, dropout, tie_encoder=None, bias=True):
        super().__init__()
        self.output_dp = RNNDropout(dropout)
        self.decoder = nn.Linear(hidden_sz, output_sz, bias=bias)
        if bias:
            self.decoder.bias.data.zero_()
        if tie_encoder:
            self.decoder.weight = tie_encoder.weight
        else:
            nn.init.kaiming_uniform_(self.decoder.weight)

    def forward(self, inputs):
        raw_outputs, outputs = inputs
        output = self.output_dp(outputs[-1])
        decoded = self.decoder(output)
        return decoded, outputs, raw_outputs


class AWDLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, padding_idx,
                 hidden_drop=0.2, input_drop=0.6, embeddings_drop=0.1, weight_drop=0.5):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.embeddings_dropout = EmbeddingsWithDropout(self.embeddings, embeddings_drop)

        self.rnn = AWDLSTM(embedding_size, hidden_size, num_layers, weight_drop, hidden_drop)

        self.embeddings.weight.data.uniform_(-0.1, 0.1)

        self.input_dropout = RNNDropout(input_drop)

    def forward(self, texts):
        embedded = self.input_dropout(self.embeddings_dropout(texts))
        raw_outputs, outputs = self.rnn(embedded)
        return raw_outputs, outputs


class EncDecLanguageModel(nn.Module):
    def __init__(self, vocab_sz, emb_sz=300, hidden_sz=300, output_sz=1, dropout=0.2,
                 pad_idx=1, num_layers=2):
        super().__init__()
        self.dps = dps = np.array([0.1, 0.15, 0.25, 0.02, 0.2]) * dropout
        self.encoder = AWDLSTMEncoder(vocab_sz, emb_sz, hidden_sz, num_layers, pad_idx,
                                      *dps[:-1])
        self.decoder = LinearDecoder(hidden_sz, vocab_sz, dps[-1], tie_encoder=self.encoder.embeddings)
        return

    def forward(self, xb):
        output_enc = self.encoder(xb)
        output_dec = self.decoder(output_enc)
        return output_dec
