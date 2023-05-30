from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
RESNET18_EMB_DIM = 1000
GPT2_HIDDEN_DIM = 768
device = "mps"

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


class hacky_embedding(torch.nn.Module):
    def __init__(self,gpt2,batch_size):
        super(hacky_embedding, self).__init__()
        self.gpt2_emb = gpt2.transformer.wte
        self.image_embeddings = torch.tensor([]) #img embeddings will be indexed by "idx of token in sequence":torch.tensor(embedding) 
    
    def forward(self,sequence):
        #embedd passed sequence
        embb = self.gpt2_emb.forward(sequence)
        for batch in range(self.image_embeddings.shape[0]): 
            embb[batch,0:self.image_embeddings.shape[1]] = nn.Parameter(self.image_embeddings[0],requires_grad=False)
        
        return embb

class caption_generator(torch.nn.Module):
    def __init__(self,gpt2,resnet18,tokens_per_img,batch_size):
        super(caption_generator, self).__init__()
        self.gpt2 = gpt2
        self.gpt2.requires_grad = False
        self.config = self.gpt2.config
        self.hacky_embedding = hacky_embedding(gpt2,batch_size).train()
        self.gpt2.transformer.wte = self.hacky_embedding #replace embedding layer with our own
    
        self.resnet18 = resnet18
        self.resnet18.requires_grad = False
        self.TOKENS_PER_IMG = tokens_per_img

        """
        #layers that convert the img features into embedding 
        self.imgfeat_to_gpt2emb = nn.Sequential(
            nn.Linear(RESNET18_EMB_DIM,10000),
            nn.ReLU(),
            nn.Linear(10000,500*tokens_per_img),
            nn.Linear(500*tokens_per_img,tokens_per_img*GPT2_HIDDEN_DIM),
            nn.ReLU(), #should I have this layer here? What is the range of values in gpt2 emb layer?
        )
        """
        #instead of FC for translating attempt selfattention block
        
        self.imgfeat_to_gpt2emb = AttentionBlock(RESNET18_EMB_DIM, RESNET18_EMB_DIM, RESNET18_EMB_DIM, tokens_per_img*GPT2_HIDDEN_DIM, 4,int(RESNET18_EMB_DIM/2))
        self.imgfeat_to_gpt2emb2 = AttentionBlock(tokens_per_img*GPT2_HIDDEN_DIM, tokens_per_img*GPT2_HIDDEN_DIM, tokens_per_img*GPT2_HIDDEN_DIM, tokens_per_img*GPT2_HIDDEN_DIM, 4,int(tokens_per_img*GPT2_HIDDEN_DIM/2))
        self.imgfeat_to_gpt2emb3 = AttentionBlock(tokens_per_img*GPT2_HIDDEN_DIM, tokens_per_img*GPT2_HIDDEN_DIM, tokens_per_img*GPT2_HIDDEN_DIM, tokens_per_img*GPT2_HIDDEN_DIM, 4,int(tokens_per_img*GPT2_HIDDEN_DIM/2))

    def forward(self,img,train=True,caption=None):
        """
        Forward function of the caption_generator model class.

        Input: img (torch.tensor)
            params: train, caption
                if train=True a caption is expected and will train the model to predict that
                if train=False no caption is expected and the model will generate a new one.
                Note: caption should already come tokenized
        Output: 
                if train=True gpt2 losses for each token and generated caption(tokenized)
                if train=False generated caption(tokenized)

        """
        #extract features with resnet:
        features = self.resnet18(img)

        #generate gpt2_embeddings
        concat_embs1 = self.imgfeat_to_gpt2emb(features,features,features)
        concat_embs2 = self.imgfeat_to_gpt2emb2(concat_embs1,concat_embs1,concat_embs1)
        concat_embs2 += concat_embs1
        concat_embs = self.imgfeat_to_gpt2emb3(concat_embs2,concat_embs2,concat_embs2)
        concat_embs += concat_embs2
        
        img_emb = concat_embs[:,0:(1)*GPT2_HIDDEN_DIM].unsqueeze(1) #axis0 is batch
        for i in range(1,self.TOKENS_PER_IMG):
            extracted_emb = concat_embs[:,i*GPT2_HIDDEN_DIM:(i+1)*GPT2_HIDDEN_DIM]
            extracted_emb  = extracted_emb.unsqueeze(1) #create sequence dimension
            img_emb = torch.cat((img_emb,extracted_emb),axis=1) #axis0 is batch
        
        
        #same image embeddings to the hacky embedding layer
        self.hacky_embedding.image_embeddings = img_emb
        #generate a random sequence of tokens for each emb in image emb
        tokens = torch.zeros((img_emb.shape[0],img_emb.shape[1]),dtype=int).to(device)
        ##CAPTION TRAINING
        if train: #at training we feed a caption
            assert caption is not None, "No caption passed! ᑫ⇀ᨓ↼ᑷ"
            #add caption tokens to the already embedded image sequence
            tokens = torch.cat((tokens,caption),axis=1)

        input = {}
        input["input_ids"] = tokens
        input["attention_mask"] = torch.ones((tokens.shape[0],tokens.shape[1]),dtype=int).to(device)

        output = self.gpt2.forward(**input)
        logits = output["logits"]
        caption_logits = logits[:,img_emb.shape[1]:,:]
        #return only hidden states for tokens (in seq lenght ) representing the caption
        return caption_logits
