from models_v3 import desperate_transformer
from dataset import ImageCaptionDataset
import pickle
from torch.utils.data import DataLoader
import wandb
import yaml
from typing import Dict
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from icecream import ic

import numpy as np

import os

ic.disable()

def generate_caption(model,features):
    features = features.to(device)
    tok = torch.tensor([[1]]).to(device)
    caption = "bos"
    for i in range(40):
        out = model(features,tok)
        out = nn.functional.softmax(out,dim=-1)
        gen_word = torch.multinomial(out[0,0], 1)
        caption += " "+vocabulary[gen_word]
        tok = gen_word.unsqueeze(0)
    
    return caption


###SCRIPT CONFIG!####
device = "cuda" #r u running cuda my boy? or mps? :D
num_epochs = 100
batch_size = 64
print("Have you runned wandb login?? OK. go aheadd...")
#####################

#convert to dictionary a yaml 
def nested_dict(original_dict):
    nested_dict = {}
    for key, value in original_dict.items():
        parts = key.split(".")
        d = nested_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return nested_dict

#load datasets
with open('/home/xnmaster/dlnn-project_ia-group_10/dataset/train_dataset.pkl', 'rb') as inp:
    train_dataset = pickle.load(inp)
with open('/home/xnmaster/dlnn-project_ia-group_10/dataset/val_dataset.pkl', 'rb') as inp:
    val_dataset = pickle.load(inp)
#with open('/Users/josepsmachine/Documents/UNI/DL/dlnn-project_ia-group_10/dataset/debug_dataset.pkl', 'rb') as inp:
#    debug_dataset = pickle.load(inp)

#create dataloaders

#debug_dataloader = DataLoader(debug_dataset, batch_size=1, shuffle=False)

#load word2vec pretrained embedding layer
word2vec_emb = api.load('word2vec-google-news-300')
word2vec_emb = torch.FloatTensor(word2vec_emb.vectors)

#import vocabulary to word2vec indexes that we know
with open('vocabidx2word2vecidx.pkl', 'rb') as inp:
    vocabidx2word2vecidx = pickle.load(inp)

with open('vocabulary.pkl', 'rb') as inp:
    vocabulary = pickle.load(inp)

word2vec_emb = word2vec_emb[vocabidx2word2vecidx]

word2vec_emb = nn.Embedding.from_pretrained(word2vec_emb)
word2vec_emb.requires_grad_ = False #freeze word2vec embeddding layer

#define cross entropy loss function
cross_entrop = nn.CrossEntropyLoss()

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

#emb_layer = word2vec_emb
#else:
    #the learnt embedding layer will have the same vocab size as word2vec for comparaison reasons.
emb_layer = nn.Embedding(num_embeddings=word2vec_emb.weight.shape[0], embedding_dim=100)
#load universal sentence encoder to define our own loss function
sntc_enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = desperate_transformer(512, emb_layer, 100, 1)
def sentence_similarity(caption,pred):
    embeddings = sntc_enc.encode([caption, pred])
    similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))
    return similarity

#print(config["see_once"])
#count_parameters(model)

model.to(device)

step_size = 1  # Number of epochs before adjusting the learning rate
gamma =  0.00001
def train(model):
    with wandb.init(project="dl2023_imagecaptioning", entity = "dl2023team") as run:
        
        run.name = f"LSTM&miniTransformer_v3"
        
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)

        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        ############
        #Train loop:
        ############

        best_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            training_losses = [] # renamed from epoch_losses
            progress_bar = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            for batch,(X1,X2,caption) in progress_bar:
            #for batch,(X1,X2) in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                X1 = X1.to(device) 
                X2 = X2.to(device)
                out = model(X1,X2)
                #ic(out.shape)
                ref = X2
                #change ref to join batch and sequence dim
                ref = ref.reshape(ref.shape[0]*ref.shape[1])


                #same for the out logits
                #change ref to join batch and sequence dim
                out = out.reshape(out.shape[0]*out.shape[1],out.shape[2])
                
                loss = cross_entrop(out,ref)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
                optimizer.step()    
                training_losses.append(loss.item())
                progress_bar.set_postfix({'Batch Loss': loss.item()})
            
            scheduler.step()

            average_training_loss = sum(training_losses) / len(training_losses) # renamed from avg_loss
            wandb.log({'Train_Epoch_Loss': average_training_loss})

            if average_training_loss < best_loss:
                best_loss = average_training_loss
                torch.save(model.state_dict(), "LSTM&miniTransformer_v1.pt")
                wandb.save(f'LSTM&miniTransformer_v1.pt')
                #print(f"Model saved at {'{wandb.run.id}_LSTM&resnet18.pt'}")
        
        wandb.finish()

train(model)

