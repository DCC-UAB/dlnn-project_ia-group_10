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
    def __init__(self, input_dim, vocab_size,embedding_dim, hidden_dim, n_heads):
        super(desperate_transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.5)  
        

        self.f_extr1 = AttentionBlock(input_dim, input_dim, input_dim, hidden_dim, n_heads,600)
        self.f_extr2 = AttentionBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim,n_heads,600)
        self.f_extr3 = AttentionBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim,n_heads,600)
        

        self.seq_mod1 = AttentionBlock(hidden_dim+35,hidden_dim+35,hidden_dim+35,hidden_dim,n_heads,600)
        self.seq_mod2 = AttentionBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim,n_heads,600)
        self.seq_mod3 = AttentionBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim,n_heads,600)
        

        self.LM_FC = nn.Linear(in_features=hidden_dim, out_features=self.embedding.weight.shape[0])
        
    
    
    def forward(self, x, x2):
        #extract image features
        o1 = self.f_extr1(x,x,x)
        o2 = self.f_extr2(o1,o1,o1) + o1
        featurs = self.f_extr3(o2,o2,o2) + o2 + o1
        
        featurs = featurs.unsqueeze(1) #create seq dim
        featurs = featurs.repeat(1,35,1) #create one for each token
        
        ic(featurs.shape)
        #first token to pass:
        seq = torch.tensor([1]).to("mps")
        #repeat initial token along batch dimension
        seq = seq.repeat(x.shape[0],1)
        ic(seq.shape)
        caption = torch.tensor([]).to("mps")
        ic(caption.shape)

        prob_distribs = torch.tensor([]).to("mps")
        for i in range(x2.shape[1]): #one token at a time, as many as caption(constant max value)
            #pass first token through transformer
            featurs_pass = featurs[:,:i+1,:]
            if i == 0: #the slicing will remove seq dim so add it again
                featurs_pass.unsqueeze(1)
            ic(featurs_pass.shape)
            ic(self.embedding(seq).shape)
            pos_enc = torch.zeros(35)
            pos_enc[i] = 1
            pos_enc = pos_enc.repeat(x2.shape[0],featurs_pass.shape[1],1).to("mps")
            ic(pos_enc.shape)
            emb_in = torch.cat((self.embedding(seq),pos_enc),axis=-1)
            feat_in = torch.cat((featurs_pass,pos_enc),axis=-1)
            o1 = self.seq_mod1(emb_in,feat_in,emb_in)
            o2 = self.seq_mod2(o1,o1,o1) + o1
            n_tok  = self.seq_mod3(o2,o2,o2) + o2 + o1
            n_tok = self.LM_FC(n_tok)
            ic(n_tok.shape)
            #find new token
            n_tok = nn.functional.softmax(n_tok[:,-1],dim=-1).unsqueeze(1) #normalize prob distr of last token
            ic(prob_distribs.shape)
            ic(n_tok.shape)
            prob_distribs = torch.cat((prob_distribs,n_tok),dim=1)
            ic(prob_distribs.shape)
            
            #get token for each element in batch
            new_tokens = []
            for i in range(n_tok.shape[0]):
                b_tok = n_tok[i]
                b_tok = torch.multinomial(b_tok, 1)
                new_tokens.append(b_tok[0].item())
            
            new_tokens = torch.tensor(new_tokens).unsqueeze(1).to("mps") #create seq dim
            ic(new_tokens.shape)
            seq = torch.cat((seq,new_tokens),dim=1)
            ic(seq.shape)

        return prob_distribs
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def init_hidden(self):
        " Initialize the hidden state of the RNN to zeros"
        h = nn.Parameter(torch.zeros(self.n_layers, 1, self.hidden_dim)).to("mps")
        c = nn.Parameter(torch.zeros(self.n_layers, 1, self.hidden_dim)).to("mps")
        return h, c