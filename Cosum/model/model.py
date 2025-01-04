import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerDecoder, TransformerEncoderLayer, TransformerEncoder
from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention, PositionalEncoding, MultiHeadAttention
from .utils import get_mask
from torch.functional import F


class Encoder(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_size,
                 device,
                 padding_idx=0,
                 embeddings=None,
                 dropout=0.5):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx)

        self._rnn_dropout = RNNDropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(self.embedding_dim,
                                        self.hidden_size)

        # self.encode = PositionalEncoding(embedding_dim)

        self._attention = SoftmaxAttention()

        self.attention = MultiHeadAttention(embedding_dim, num_heads=8)

        self._projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size, self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(self.hidden_size,
                                           self.hidden_size)

        # self.composition = PositionalEncoding(self.hidden_size)

        self.pro = nn.Sequential(nn.Linear(2 * self.hidden_size, self.embedding_dim),
                                 nn.ReLU())

    def forward(self, p1, p1_lengths, p2, p2_lengths):

        p1_mask = get_mask(p1, p1_lengths).to(self.device)
        p2_mask = get_mask(p2, p2_lengths).to(self.device)

        embedded_p1 = self._word_embedding(p1)
        embedded_p2 = self._word_embedding(p2)

        embedded_p1 = self._rnn_dropout(embedded_p1)
        embedded_p2 = self._rnn_dropout(embedded_p2)

        encoded_p1 = self._encoding(embedded_p1, p1_lengths).to(self.device)
        encoded_p2 = self._encoding(embedded_p2, p2_lengths).to(self.device)

        # encoded_p1 = self.encode(embedded_p1)
        # encoded_p2 = self.encode(embedded_p2)

        attended_p1, attended_p2 = self._attention(encoded_p1, encoded_p2, p1_mask, p2_mask)

        # attended_p1 = self.attention(encoded_p1)
        # attended_p2 = self.attention(encoded_p2)

        enhanced_p1 = torch.cat([encoded_p1, attended_p1,
                                 encoded_p1 - attended_p1,
                                 encoded_p1 * attended_p1],
                                dim=-1)
        enhanced_p2 = torch.cat([encoded_p2, attended_p2,
                                 encoded_p2 - attended_p2,
                                 encoded_p2 * attended_p2],
                                dim=-1)

        projected_p1 = self._projection(enhanced_p1)
        projected_p2 = self._projection(enhanced_p2)

        projected_p1 = self._rnn_dropout(projected_p1)
        projected_p2 = self._rnn_dropout(projected_p2)

        v_ai = self._composition(projected_p1, p1_lengths)
        v_bj = self._composition(projected_p2, p2_lengths)


        # v_ai = self.composition(projected_p1)
        # v_bj = self.composition(projected_p2)

        v = torch.cat([v_ai, v_bj], dim=1)

        # v = self._rnn_dropout(v)

        # v = self.pro(v)

        return v

class ESCG(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 device,
                 dropout,
                 padding_idx=0,
                 embeddings=None):
        super(ESCG, self).__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device
        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx)

        self._encoder = Encoder(vocab_size,
                                embedding_dim,
                                hidden_size,
                                embeddings=embeddings,
                                device=device,
                                dropout=dropout,
                                padding_idx=padding_idx)

        self.encoderlayer = TransformerEncoderLayer(self.embedding_dim,
                                                    nhead=6,
                                                    batch_first=True)
        self.encoder = TransformerEncoder(self.encoderlayer,
                                          num_layers=2)

        self._decodelayer = TransformerDecoderLayer(self.embedding_dim,
                                                    nhead=6,
                                                    batch_first=True)

        self._decode = TransformerDecoder(self._decodelayer,
                                          num_layers=2)

        self.generator = nn.Linear(self.embedding_dim,
                                   self.vocab_size)

    def forward(self, p1, p1_lengths, p2, p2_lengths, trg):

        memory = self._encoder(p1, p1_lengths, p2, p2_lengths)

        # memory = self.encoder(memory)

        tgt_emb = self._word_embedding(trg)

        tgt_seq_len = trg.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

        cit = self._decode(tgt_emb, memory, tgt_mask)

        return F.log_softmax(self.generator(cit), dim=-1)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=self.device), diagonal=1)
        return mask
