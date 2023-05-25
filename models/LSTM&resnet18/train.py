from models import LSTMModel , LSTMModel_seeOnce
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

from tqdm import tqdm
from icecream import ic

import numpy as np

import os

os.environ["TOKENIZERS_PARALLELISM"] = "False"


###SCRIPT CONFIG!####
device = "cuda" #r u running cuda my boy? or mps? :D
num_epochs = 50
batch_size = 300
COSINE_SIM_IMPORTANCE = 0.8
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
#load datasets
with open('/home/xnmaster/dlnn-project_ia-group_10/dataset/train_dataset.pkl', 'rb') as inp:
    train_dataset = pickle.load(inp)
with open('/home/xnmaster/dlnn-project_ia-group_10/dataset/val_dataset.pkl', 'rb') as inp:
    val_dataset = pickle.load(inp)
#with open('/Users/josepsmachine/Documents/UNI/DL/dlnn-project_ia-group_10/dataset/debug_dataset.pkl', 'rb') as inp:
#    debug_dataset = pickle.load(inp)

#create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#debug_dataloader = DataLoader(debug_dataset, batch_size=1, shuffle=False)

#load hyperparamaters to do grid search on
#setup wandb stuff
with open('hyperparams.yaml', 'r') as stream:
    try:
        sweep_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

#create sweep
sweep_id = wandb.sweep(sweep_config, project="energy_project_uab")

#load word2vec pretrained embedding layer
word2vec_emb = api.load('word2vec-google-news-300')
word2vec_emb = torch.LongTensor(word2vec_emb.vectors)

#import vocabulary to word2vec indexes that we know
with open('vocabidx2word2vecidx.pkl', 'rb') as inp:
    vocabidx2word2vecidx = pickle.load(inp)

with open('vocabulary.pkl', 'rb') as inp:
    vocabulary = pickle.load(inp)

word2vec_emb = word2vec_emb[vocabidx2word2vecidx]

word2vec_emb = nn.Embedding.from_pretrained(word2vec_emb)
word2vec_emb.requires_grad_ = False #freeze word2vec embeddding layer

#load universal sentence encoder to define our own loss function
sntc_enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#define cross entropy loss function
cross_entrop = nn.CrossEntropyLoss()

def train(config: Dict = None):
    with wandb.init(config, project="dl2023_imagecaptioning", entity = "dl2023team") as run:
        #Set run name
        config = wandb.config
        config = nested_dict(config)

        ##set name
        if config["see_once"]:
            run.name = f"LSTM_seeOnce_{wandb.run.id}"
        else:
            run.name= f"LSTM_residuals_{wandb.run.id}"
        
        print("name set to: ",wandb.run.name)
        #set embedding layer
        if config["embedding_layer"] == "word2vec":
                emb_layer = word2vec_emb
        else:
            #the learnt embedding layer will have the same vocab size as word2vec for comparaison reasons.
            emb_layer = nn.Embedding(num_embeddings=word2vec_emb.weight.shape[0], embedding_dim=config["hidden_size"])

        if config["see_once"]: #if we want the model with residual at each step or not
            model = LSTMModel_seeOnce(input_dim=512,embedding_layer=emb_layer,hidden_dim=config["hidden_size"],n_layers=config['num_layers'])
        else:
            model = LSTMModel(input_dim=512,embedding_layer=emb_layer,hidden_dim=config["hidden_size"],n_layers=config['num_layers'])
        
        #print(config["see_once"])
        #count_parameters(model)
        
        model.init_weights()
        model.to(device)

        #define optimizer for this run
        optimizer_config = config["optimizer"]
        if optimizer_config["type"] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = optimizer_config['lr'])
        
        #define loss for this run
        if config["loss_funct"] == "crossentropy":
            def loss_funct(ref,pred): 
                ref = ref.type(torch.LongTensor).to(device)  # Convert ref to torch.LongTensor and move to device
                pred = pred.type(torch.LongTensor).to(device)  # Convert pred to torch.LongTensor and move to device
                loss = cross_entrop(pred,ref)
                return loss
        else:
            def loss_funct(ref,pred):
                ref = ref.type(torch.LongTensor).to(device)  # Convert ref to torch.LongTensor and move to device
                pred = pred.type(torch.LongTensor).to(device)  # Convert pred to torch.LongTensor and move to device
                loss1 = cross_entrop(pred,ref)

                pred_keys = torch.argmax(pred,axis=-1)
                
                pred_sntc = ""
                target_sntc = ""
                
                for batch_ref,batch_pred in zip(ref,pred_keys):
                    pred_sntc += " " + vocabulary[batch_pred.item()]
                    target_sntc += " " + vocabulary[batch_ref.item()]

                #ic(caption)
                #ic(target_sntc)
                #ic(pred_sntc)

                embeddings = sntc_enc.encode([pred_sntc, target_sntc])
                similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))
                loss = 1/(similarity[0][0])*COSINE_SIM_IMPORTANCE + loss1*(1-COSINE_SIM_IMPORTANCE)
                return loss

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
                
                X1 = X1.type(torch.LongTensor) .to(device)
                X2 = X2.type(torch.LongTensor).to(device)
                out,h,c = model(X1,X2)
                
                ref = X2
                #change ref to join batch and sequence dim
                ref = ref.view(ref.shape[0]*ref.shape[1])


                #same for the out logits
                #change ref to join batch and sequence dim
                out = out.view(out.shape[0]*out.shape[1],out.shape[2])
                
                loss = loss_funct(ref,out)
                loss.backward()
                optimizer.step()    
                training_losses.append(loss.item())
                progress_bar.set_postfix({'Batch Loss': loss.item()})

            average_training_loss = sum(training_losses) / len(training_losses) # renamed from avg_loss
            wandb.log({'Train_Epoch_Loss': average_training_loss})

            model.eval()  
            with torch.no_grad():  
                validation_losses = [] # renamed from val_losses
                for X1,X2,caption in tqdm(val_dataloader, desc='Validation'):
                    X1 = X1.type(torch.LongTensor).to(device)
                    X2 = X2.type(torch.LongTensor).to(device)

                    out,h,c = model(X1,X2)
                    ref = X2

                    ref = ref.view(ref.shape[0]*ref.shape[1])
                    out = out.view(out.shape[0]*out.shape[1],out.shape[2])
                    
                    loss = loss_funct(ref,out)

                    validation_losses.append(loss.item())

                average_validation_loss = sum(validation_losses) / len(validation_losses) # renamed from avg_val_loss
                wandb.log({'Validation_Epoch_Loss': average_validation_loss})

            if average_training_loss < best_loss:
                best_loss = average_training_loss
                torch.save(model.state_dict(), f'{wandb.run.name}.pt')
                wandb.save(f'{wandb.run.name}.pt')
                print(f"Model saved at {'{run.name}.pt'}")

        wandb.finish()


wandb.agent(sweep_id, function=train)

