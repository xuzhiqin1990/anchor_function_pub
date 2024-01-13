import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import time
from tqdm import tqdm



class myLSTM(nn.Module):
    def __init__(self, args, device):
        super(myLSTM, self).__init__()

        self.device = device

        self.tgt_emb = nn.Embedding(args.vocab_size, args.d_model)
        self.lstm_layer = torch.nn.LSTM(input_size = args.d_model, hidden_size = args.d_model, 
                                        num_layers = args.n_layers, batch_first=True)
        self.projection = nn.Linear(4*args.d_model, args.vocab_size)


        # self.decoder = Decoder()
        # self.projection = nn.Linear(d_model, vocab_size)

    
    def forward(self, dec_inputs):
        """
        dec_inputs: [batch_size, tgt_len]
        """

        hidden_state = self.tgt_emb(dec_inputs)

        hidden_state, _ = self.lstm_layer(hidden_state)

        prob = self.projection(hidden_state)

        return prob.view(-1, prob.size(-1)), None
        # return prob, None

    # def forward(self,dec_inputs):
    #     """
    #     dec_inputs: [batch_size, tgt_len]
    #     """

    #     dec_outputs, dec_self_attns = self.decoder(dec_inputs)
    #     dec_logits = self.projection(dec_outputs)
    #     return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns
    

    def greedy_decoder(self,dec_input):

        projected, _ = self.forward(dec_input)

        projected = projected[-1,:].argmax()
        next_word = projected.item() 

        return next_word


    def test(self,sentence):
        dec_input = torch.tensor(sentence, dtype=torch.long, device=self.device).unsqueeze(0)

        output = self.greedy_decoder(dec_input)

        return output

