import torch
import torch.utils.data as Data
from torch import nn
import numpy as np


def get_attn_pad_mask(seq_q, seq_k):
    '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq, device):
    '''
        seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

def attn_mask(X_input, device):
    '''
        X_input: [batch_size, tgt_len]
    '''
    dec_self_attn_pad_mask = get_attn_pad_mask(X_input, X_input) # [batch_size, tgt_len, d_model] 遮挡padding部分
    dec_self_attn_subsequence_mask = get_attn_subsequence_mask(X_input, device) # [batch_size, tgt_len, d_model] 遮挡未来时刻的词
    # 两个mask之和只要有一个为1的地方，就为1
    dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0) # [batch_size, tgt_len, d_model] 

    return dec_self_attn_mask


def get_attn_pad_mask(seq_q, seq_k):
    '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq, device):
    '''
        seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

class ScaledDotProductAttention(nn.Module):
        def __init__(self):
            super(ScaledDotProductAttention, self).__init__()

        def forward(self, Q, K, V, attn_mask, d_k):
            '''
                Q: [batch_size, n_heads, len_q, d_k]
                K: [batch_size, n_heads, len_k, d_k]
                V: [batch_size, n_heads, len_v(=len_k), d_v]
                attn_mask: [batch_size, n_heads, seq_len, seq_len]
            '''
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
                d_k)  # scores : [batch_size, n_heads, len_q, len_k]
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

            attn = nn.Softmax(dim=-1)(scores)
            context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
            return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()

        self.n_head = args.n_heads
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.d_model = args.d_model
    
        self.W_Q = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_K = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_V = nn.Linear(args.d_model, args.d_v * args.n_heads, bias=False)
        self.fc = nn.Linear(args.n_heads * args.d_v, args.d_model, bias=False)
        self.layernorm = nn.LayerNorm(args.d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask, self.d_k)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return self.layernorm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.d_model, args.d_feedforward, bias=False),
            nn.ReLU(),
            nn.Linear(args.d_feedforward, args.d_model, bias=False)
        )
        self.layernorm=nn.LayerNorm(args.d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual) # [batch_size, seq_len, d_model]

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, dec_inputs, dec_self_attn_mask):
        '''
            dec_inputs: [batch_size, tgt_len, d_model]
            dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        # Attention层
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        # 非线性层
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn


class Decoder(nn.Module):
    def __init__(self, args, device):
        super(Decoder, self).__init__()
        self.device = device
        self.tgt_emb = nn.Embedding(args.vocab_size, args.d_model)
        self.pos_emb = nn.Embedding(args.max_pos, args.d_model)
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])

    def forward(self, hidden_state, dec_self_attn_mask):
        '''
            dec_inputs: [batch_size, tgt_len]
        '''
        dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model]
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            hidden_state, dec_self_attn = layer(hidden_state, dec_self_attn_mask)
   
            dec_self_attns.append(dec_self_attn)

        return hidden_state, dec_self_attns


class Embedding(nn.Module):
    def __init__ (self, args, device):
        super(Embedding, self).__init__()
        self.device = device
        self.tgt_emb = nn.Embedding(args.vocab_size, args.d_model)
        self.pos_emb = nn.Embedding(args.max_pos, args.d_model)

    def forward(self, X_input):
        seq_len = X_input.size(1)
        pos = torch.arange(seq_len, dtype = torch.long, device = self.device)
        pos = pos.unsqueeze(0).expand_as(X_input)

        self.tgt = self.tgt_emb(X_input)
        self.pos = self.pos_emb(pos)
        self.emb = self.tgt + self.pos

        return self.emb


class myGPT_specific(nn.Module):
    def __init__(self, args, device):
        super(myGPT_specific, self).__init__()

        self.device = device
        self.embedding = Embedding(args, device)
        self.decoder = Decoder(args, device)
        self.projection = nn.Linear(args.d_model, args.vocab_size)


    def forward(self, X_input):
        """
            dec_inputs: [batch_size, tgt_len]
        """
        hidden_state = self.embedding(X_input)

        self.dec_self_attn_mask = attn_mask(X_input, self.device)

        hidden_state, dec_self_attns = self.decoder(hidden_state, self.dec_self_attn_mask)
        
        self.dec_logits = self.projection(hidden_state)
        
        return self.dec_logits.view(-1, self.dec_logits.size(-1)), dec_self_attns
    

    def greedy_decoder(self,dec_input):

        projected, _ = self.forward(dec_input)

        projected = projected[-1,:].argmax()
        next_word = projected.item() 

        return next_word


    def test(self, sentence):
        dec_input = torch.tensor(sentence, dtype=torch.long, device=self.device).unsqueeze(0)

        output = self.greedy_decoder(dec_input)

        return output




