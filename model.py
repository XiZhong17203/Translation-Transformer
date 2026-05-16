import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = yaml.load(open(os.path.join(BASE_DIR, 'config.yaml'), 'r'), Loader=yaml.FullLoader)

# TODO:Position Encoding
class PositionEncoding(nn.Module):
    '''
    tensor: [batch_size, seq_len, d_model]
    '''
    def __init__(self, d_model=config['d_model'], max_len=config['max_len']):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# TODO:Input Embedding
class InputEmbedding(nn.Module):
    '''
    input: [batch_size, seq_len]\\
    output: [batch_size, seq_len, d_model]
    '''
    def __init__(self, cn_vocab_size=config['cn_vocab_size'], en_vocab_size=config['en_vocab_size'], d_model=config['d_model']):
        super().__init__()
        self.src_embedding = nn.Embedding(en_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(cn_vocab_size, d_model)
    def forward(self, src, tgt):
        src_emb = self.src_embedding(src) * torch.sqrt(torch.tensor(config['d_model'], dtype=torch.float))
        tgt_emb = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(config['d_model'], dtype=torch.float))
        return src_emb, tgt_emb

# TODO:Multi-Head Attention
class MultiHeadAttn(nn.Module):
    '''
    tensor: [batch_size, seq_len, d_model]\\
    mask: [batch_size, seq_len, seq_len]\\
    dq = dk = dv = d_model // num_heads
    '''
    def __init__(self, d_model=config['d_model'], num_heads=config['nhead']):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # batch_size, seq_len, d_model -> batch_size, num_heads, seq_len, head_dim
        Q = self.Wq(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # batch_size, num_heads, seq_len, head_dim -> batch_size, num_heads, seq_len, seq_len
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # batch_size, num_heads, seq_len, softmax(seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        # batch_size, num_heads, seq_len, head_dim -> batch_size, seq_len, d_model
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.Wo(attn_output)
        return attn_output

# TODO:Feed-Forward Networks
class ffn(nn.Module):
    '''
    tensor: [batch_size, seq_len, d_model]
    '''
    def __init__(self, d_model=config['d_model'], d_ff=config['dim_feedforward']):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# TODO:Encoder Layer
class EncoderLayer(nn.Module):
    '''
    tensor: [batch_size, seq_len, d_model]\\
    mask: [batch_size, seq_len, seq_len]
    '''
    def __init__(self, d_model=config['d_model'], num_heads=config['nhead'], d_ff=config['dim_feedforward'], dropout=config['dropout']):
        super().__init__()
        self.self_attn = MultiHeadAttn(d_model, num_heads)
        self.ffn = ffn(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(x, x, x, mask))
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x

# TODO:Decoder Layer
class DecoderLayer(nn.Module):
    '''
    tensor: [batch_size, seq_len, d_model]\\
    mask: [batch_size, seq_len, seq_len]
    '''
    def __init__(self, d_model=config['d_model'], num_heads=config['nhead'], d_ff=config['dim_feedforward'], dropout=config['dropout']):
        super().__init__()
        self.masked_attn = MultiHeadAttn(d_model, num_heads)
        self.cross_attn = MultiHeadAttn(d_model, num_heads)
        self.ffn = ffn(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # tgt_mask: padding + subsequent mask
        a, b = tgt_mask
        tgt_mask = a & b
        x = x + self.dropout(self.masked_attn(x, x, x, tgt_mask))
        x = self.norm1(x)
        # q:x k,v:encoder
        x = x + self.dropout(self.cross_attn(x, enc_output, enc_output, src_mask))
        x = self.norm2(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm3(x)
        return x

# TODO:Transformer
class Transformer(nn.Module):
    def __init__(self, num_encoder_layers=config['num_encoder_layers'], num_decoder_layers=config['num_decoder_layers'], d_model=config['d_model'], num_heads=config['nhead'], d_ff=config['dim_feedforward'], dropout=config['dropout']):
        super().__init__()
        self.input_embedding = InputEmbedding()
        self.position_encoding = PositionEncoding()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)])
        self.output_linear = nn.Linear(d_model, config['cn_vocab_size'])
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb, tgt_emb = self.input_embedding(src, tgt)
        src_emb = self.dropout(self.position_encoding(src_emb))
        tgt_emb = self.dropout(self.position_encoding(tgt_emb))
        enc_output = src_emb
        tgt_output = tgt_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
            tgt_output = layer(tgt_output, tgt_mask[0])
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        # dec_output: [batch_size, seq_len, d_model] -> output: [batch_size, seq_len, cn_vocab_size]
        output = self.output_linear(dec_output)
        # Add encoder output for SimCLR
        return output, (enc_output, tgt_output)

# TODO:Add SimCLR
# class SimCLR(nn.Module):
#     def __init__(self, d_model=config['d_model'], proj_dim=128, temperature=0.1):
#         super().__init__()
        
#         self.temperature = temperature
#         self.src_projector = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, proj_dim)
#         )
        
#         self.tgt_projector = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, proj_dim)
#         )
    
#     def pooling(self, features, mask):
#         # features: [batch_size, seq_len, d_model], mask: [batch_size, seq_len]
#         # print(features.size(), mask.size())
#         masked_features = features * mask.unsqueeze(-1)
#         pooled = masked_features.sum(dim=1) / mask.sum(dim=1, keepdim=True)
#         return pooled
    
#     def forward(self, feature1, feature2, mask1, mask2):
#         # feature1, feature2: [batch_size, seq_len, d_model], mask1, mask2: [batch_size, 1, 1, seq_len]
#         pooled1 = self.pooling(feature1, mask1)
#         pooled2 = self.pooling(feature2, mask2)
        
#         z1 = self.src_projector(pooled1)
#         z2 = self.tgt_projector(pooled2)
        
#         z1 = F.normalize(z1, dim=1)
#         z2 = F.normalize(z2, dim=1)
        
#         logits = z1 @ z2.T / self.temperature
#         labels = torch.arange(logits.size(0)).to(logits.device)
#         loss1 = F.cross_entropy(logits, labels)
#         loss2 = F.cross_entropy(logits.T, labels)
#         loss = (loss1 + loss2) / 2
        
#         return loss