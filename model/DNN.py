import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import time
from tqdm import tqdm

class FeedForwardNet(nn.Module):
    def __init__(self, args):
        super(FeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.seq_len*args.d_model, 4*args.seq_len*args.d_model),
            nn.ReLU(),
            nn.Linear(4*args.seq_len*args.d_model, args.seq_len*args.d_model)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len * d_model]
        '''
        output = self.fc(inputs)
        return output + inputs



class myDNN(nn.Module):
    def __init__(self, args, device):
        super(myDNN, self).__init__()

        self.device = device

        self.tgt_emb = nn.Embedding(args.vocab_size, args.d_model)
        self.pos_emb = nn.Embedding(args.max_pos, args.d_model)

        self.layers = nn.ModuleList([FeedForwardNet(args) for _ in range(args.n_layers)])

        self.fnn = nn.Linear(args.seq_len * args.d_model, args.d_model)

        self.projection = nn.Linear(args.d_model, args.vocab_size)

        self.d_model = args.d_model
    
    def forward(self, dec_inputs):
        """
        dec_inputs: [batch_size, tgt_len]
        """
        seq_len = dec_inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long,device=self.device)
        pos = pos.unsqueeze(0).expand_as(dec_inputs)  # [seq_len] -> [batch_size, seq_len]
        
        # 先做正常的embedding和位置编码
        hidden_state = self.tgt_emb(dec_inputs) + self.pos_emb(pos) # [batch_size, tgt_len, d_model]

        # 将hidden_state展平
        hidden_state = hidden_state.view(-1,seq_len*self.d_model)

        # 每层将神经元数扩大4倍，ReLU后再缩小回来
        for layer in self.layers:
            hidden_state = layer(hidden_state)

        # 这两步实现了投影
        hidden_state = self.fnn(hidden_state)

        prob = self.projection(hidden_state)

        return prob, None
    

    def greedy_decoder(self, dec_input):

        prob, _ = self.forward(dec_input)

        prob = prob.squeeze(0).argmax()

        # prob = prob.max(dim=-1, keepdim=False)[1]
        next_word = prob.item() 

        return next_word


    def answer(self,sentence):
        #把原始句子的\t替换成”<sep>“
        # dec_input = [word2id.get(word,1) if word!='\t' else word2id['<sep>'] for word in sentence]
        sentence=sentence.split('/t')[0].split(',')
        print(sentence)
        sentence = list(map(int, sentence))
        dec_input = torch.tensor(sentence, dtype=torch.long, device=self.device).unsqueeze(0)

        # print(dec_input.dtype)

        output = self.greedy_decoder(dec_input).squeeze(0)
        print(output)

    def test(self,sentence):
        dec_input = torch.tensor(sentence, dtype=torch.long, device=self.device).unsqueeze(0)

        output = self.greedy_decoder(dec_input)

        return output