from torch import nn
import torch

class SelfAttention(nn.Module):
    """ self attention module"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width = x.size()
        proj_query = self.query(x).transpose(2, 1)
        proj_key = self.key(x)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x)

        out = torch.bmm(proj_value, attention)
        out = out.reshape(m_batchsize, C, width)

        return out


def main():
    attention_block = SelfAttention(64)
    input = torch.rand([4, 64, 1000])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()
