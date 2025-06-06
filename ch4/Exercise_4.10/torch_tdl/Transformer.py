import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.src_mask = None#用于存储掩码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_length)#位置编码器
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)#编码器层
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)#多个encoder_layer构成完整的Transformer编码器
        self.d_model = d_model#模型维度

        # Define a feed-forward network for the transformer output
        self.fc_out = nn.Linear(d_model, input_dim)#利用线性层将输出维度转换到和输入维度一样

    def _generate_square_subsequent_mask(self, sz):
        '''
            用于生成掩码，防止位置之间的信息泄露
        '''
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)#创建上三角矩阵用于掩码
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        print(output.shape)
        output = self.fc_out(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Parameters
input_dim = 768  # The last dimension of the input tensor
d_model = 768    # The embedding dimension for the transformer
nhead = 8        # The number of attention heads
num_encoder_layers = 6  # The number of sub-encoder-layers in the transformer
dim_feedforward = 2048  # The dimension of the feedforward network model
dropout = 0.1     # The dropout value

# Create the Transformer model
model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=dropout)

# Create a dummy input tensor with shape [batch_size, sequence_length, feature_dim]
dummy_input = torch.rand(1, 14, 768)  # batch_size = 1, sequence_length = 14, feature_dim = 768

# Extract features using the Transformer model
features = model(dummy_input)
print(features.shape)  # Should be [1, 14, 768]