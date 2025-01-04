import torch.nn as nn
import torch
import math
from .utils import sort_by_seq_lens, masked_softmax, weighted_sum
from torch.functional import F

class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.

    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """

    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.

        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).

        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch

class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.

    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """

        super(Seq2SeqEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self._encoder = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=True)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.

        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        sorted_batch, sorted_lengths, _, restoration_idx = \
            sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths.to('cpu'),
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_proj = nn.Linear(input_size, input_size)
        self.value_proj = nn.Linear(input_size, input_size)
        self.query_proj = nn.Linear(input_size, input_size)
        self.output_proj = nn.Linear(input_size, input_size)

    def forward(self, x, mask=None):
        batch_size, seq_len, input_size = x.size()
        key = self.key_proj(x).view(batch_size, seq_len, self.num_heads, input_size // self.num_heads).transpose(1,
                                                                                                                 2)  # (batch_size, num_heads, seq_len, head_size)
        value = self.value_proj(x).view(batch_size, seq_len, self.num_heads, input_size // self.num_heads).transpose(1,
                                                                                                                     2)  # (batch_size, num_heads, seq_len, head_size)
        query = self.query_proj(x).view(batch_size, seq_len, self.num_heads, input_size // self.num_heads).transpose(1,
                                                                                                                     2)  # (batch_size, num_heads, seq_len, head_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (
                    input_size // self.num_heads) ** 0.5  # (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                                                  input_size)  # (batch_size, seq_len, input_size)
        output = self.output_proj(output)
        return output


class SoftmaxAttention(nn.Module):
    """
    Attention layer taking p1 and p2 encoded by an RNN as input
    and computing the soft attention between their elements.

    The dot product of the encoded vectors in the p1 and p2 is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the p1 for each element of the p2, and
    conversely for the elements of the p1.
    """

    def forward(self, p1_batch, p2_batch, p1_mask=None,p2_mask=None):
        """
        Args:
            p1_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            p1_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            p2_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            p2_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.

        Returns:
            attended_1: The sequences of attention vectors for the
                p1 in the input batch.
            attended_p2: The sequences of attention vectors for the
                p2 in the input batch.
        """
        similarity_matrix = p1_batch.bmm(p2_batch.transpose(2, 1)
                                         .contiguous())

        # Softmax attention weights.
        p1_p2_attn = masked_softmax(similarity_matrix, p2_mask)
        p2_p1_attn = masked_softmax(similarity_matrix.transpose(1, 2)
                                    .contiguous(),
                                    p1_mask)

        attended_p1 = weighted_sum(p2_batch,
                                   p1_p2_attn,
                                   p1_mask)
        attended_p2 = weighted_sum(p1_batch,
                                   p2_p1_attn,
                                   p2_mask)

        return attended_p1, attended_p2


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
