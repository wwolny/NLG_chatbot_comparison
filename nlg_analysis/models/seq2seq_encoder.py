import torch
from torch import nn


class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(Seq2SeqEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = device

    def set_hidden_size(self, size):
        self.hidden_size = size

    def forward(self, x, hidden):
        embedded = self.embedding(x).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
