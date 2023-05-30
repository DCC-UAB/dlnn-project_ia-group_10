import torch.nn as nn
import torch.nn.init as init
from icecream import ic
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_layer, hidden_dim, n_layers, drop_prob=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.imf2lstm_h = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.imf2lstm_c = nn.Linear(in_features=input_dim, out_features=hidden_dim)

        self.embedding = embedding_layer
        
        self.lstm = nn.LSTM(input_size=self.embedding.weight.shape[1], hidden_size=hidden_dim, num_layers=n_layers, dropout=drop_prob, batch_first=True)
        self.LM_FC = nn.Linear(in_features=hidden_dim, out_features=self.embedding.weight.shape[0])

        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, y, h=None, c=None, train=True):
        #convert image features to initial hidden state of lstm
        
        h_0,c_0 = self.init_hidden(x)

        emb = self.embedding(y)
        #instead of pasing the sequence directly to lstm
        #we want to do one step at a time and pass at each step a concat of the# Iterate over the second dimension
        h = h_0
        c = c_0
        
        outs = []
        if train:
            rang = range(emb.size(1))
        else:
            rang = range(40)
            
        for i in rang:
            if train==True:
                step = emb[:, i:i+1, :]
            else:
                #use previous token to generate a new one

                if i == 0:#if this is the first token use start token bos
                    step = torch.tensor([[1]]) #we know by hand that it is 1
                    step = self.embedding(step)
                else:
                    out = self.softmax(out)
                    step = torch.multinomial(out[0,0,:], 1)
                    step = self.embedding(step.unsqueeze(0))
            
            out, (h, c) = self.lstm(step,(h,c))
            c = c_0*0.5 + c*0.5 #have last block reset memory and put in image features
            
            outs.append(out)
        
            out = torch.cat(outs, dim=1)

        out = self.LM_FC(out)

        out = self.softmax(out)#not really necessary with cross entropy but for the other yes so keeep it :)

        return out, h, c

    def init_hidden(self,x):
        #depending on the number of layers we might want to repeat the
        #passed data through 
        h = self.imf2lstm_h(x) #create layer dim
        c = self.imf2lstm_c(x) #create layer dim
        h = h.repeat(self.n_layers,1,1) #add extra dimension per layer by repeating the tensor
        c = c.repeat(self.n_layers,1,1)
        return h, c
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)


class LSTMModel_seeOnce(nn.Module):
    def __init__(self, input_dim, embedding_layer, hidden_dim, n_layers, drop_prob=0.0):
        super(LSTMModel_seeOnce, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.imf2lstm_h = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.imf2lstm_c = nn.Linear(in_features=input_dim, out_features=hidden_dim)

        self.embedding = embedding_layer
        
        self.lstm = nn.LSTM(input_size=self.embedding.weight.shape[1], hidden_size=hidden_dim, num_layers=n_layers, dropout=drop_prob, batch_first=True)
        self.LM_FC = nn.Linear(in_features=hidden_dim, out_features=self.embedding.weight.shape[0])

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, y, h=None, c=None, train=True):
        #convert image features to initial hidden state of lstm
        if train:
            h,c = self.init_hidden(x)

        emb = self.embedding(y)
        #instead of pasing the sequence directly to lstm
        #we want to do one step at a time and pass at each step a concat of the
        out, (h, c) = self.lstm(emb,(h,c))

        out = self.LM_FC(out)

        out = self.softmax(out)#not really necessary with cross entropy but for the other yes so keeep it :)

        return out, h, c

    def init_hidden(self,x):
        #depending on the number of layers we might want to repeat the
        #passed data through 
        h = self.imf2lstm_h(x) #create layer dim
        c = self.imf2lstm_c(x) #create layer dim
        h = h.repeat(self.n_layers,1,1) #add extra dimension per layer by repeating the tensor
        c = c.repeat(self.n_layers,1,1)
        return h, c
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)