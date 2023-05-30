import torch
import torch.nn as nn
from icecream import ic
import torch.nn.init as init

class AttentionBlock(nn.Module):
    def __init__(self, key_dim, val_dim, query_dim, hidden_dim, num_heads,attnFCdim):
        super(AttentionBlock, self).__init__()
        self.key_gen = nn.Sequential(
            nn.Linear(key_dim, attnFCdim),
            nn.ReLU(),
            nn.Linear(attnFCdim, attnFCdim),
            nn.ReLU(),
            nn.Linear(attnFCdim, hidden_dim),
            nn.ReLU(),
        )

        self.val_gen = nn.Sequential(
            nn.Linear(val_dim, attnFCdim),
            nn.ReLU(),
            nn.Linear(attnFCdim, attnFCdim),
            nn.ReLU(),
            nn.Linear(attnFCdim, hidden_dim),
            nn.ReLU(),
        )
        
        self.query_gen = nn.Sequential(
            nn.Linear(query_dim, attnFCdim),
            nn.ReLU(),
            nn.Linear(attnFCdim, attnFCdim),
            nn.ReLU(),
            nn.Linear(attnFCdim, hidden_dim),
            nn.Softmax(dim=-1),
        )
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
    
    def forward(self,keys,values,queries):
        key = self.key_gen(keys) #generate key with FC key is directly the embedding
        value = self.val_gen(values) #generate value with FC
        query = self.query_gen(queries) #generate query with FC
        output, _ =  self.multihead_attn(key=key, value=value, query=query)
        return output

class desperate_transformer(nn.Module):
    def __init__(self, input_dim, embedding_layer, hidden_dim, n_heads):
        super(desperate_transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = embedding_layer
        self.n_layers = 1
        self.emb_dim = self.embedding.weight.shape[1]
        
        self.lay1 = AttentionBlock(input_dim, input_dim, input_dim, self.emb_dim, n_heads,300)
        self.lay2 = AttentionBlock(self.emb_dim,self.emb_dim,self.emb_dim,self.emb_dim,n_heads,300)
        #self.lay3 = AttentionBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim,n_heads,100)

        #self.gru = nn.GRU(self.embedding.weight.shape[1], hidden_dim, batch_first=True)
        self.lstm = nn.LSTM(self.emb_dim*2, hidden_dim, batch_first=True,dropout=0)
        
        self.LM_FC = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=2000),
            nn.Linear(2000, 6000),
            nn.Linear(6000, out_features=self.embedding.weight.shape[0])
            )
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x, x2):
        #convert image features to initial hidden state of lstm
        #ic(x.shape)
        #ic(x2.shape)
        o1 = self.lay1(x,x,x)
        #ic(o1.shape)
        o2 = self.lay2(o1,o1,o1) + o1
        #ic(o2.shape)
        featurs = self.lay2(o2,o2,o2) + o2
        #ic(hidden.shape)
        featurs = featurs.unsqueeze(1)
        featurs = featurs.repeat(1,35,1)
        embeddings = self.embedding(x2)

        input  =torch.cat((featurs,embeddings),axis=-1)
        #out, hidden = self.gru(embeddings, hidden)
        h,c = self.init_hidden(x2.shape[0])
        out,(_,_) = self.lstm(input,(h,c))
        out = self.LM_FC(out)
        out = self.dropout(out)
        return out
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def init_hidden(self, batch_size):
        " Initialize the hidden state of the RNN to zeros"
        h = nn.Parameter(torch.zeros(self.n_layers, batch_size, self.hidden_dim)).to("mps")
        c = nn.Parameter(torch.zeros(self.n_layers, batch_size, self.hidden_dim)).to("mps")
        return h, c