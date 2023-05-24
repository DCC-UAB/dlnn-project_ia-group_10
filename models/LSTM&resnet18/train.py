from models import LSTMModel
from dataset import ImageCaptionDataset
import pickle
from torch.utils.data import DataLoader
import wandb
import yaml
from typing import Dict
import gensim.downloader as api
import tensorflow_hub as hub
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
device = "cpu" #r u running cuda my boy? or mps? :D
num_epochs = 80
batch_size = 1
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
with open('/Users/josepsmachine/Documents/UNI/DL/dlnn-project_ia-group_10/dataset/train_dataset.pkl', 'rb') as inp:
    train_dataset = pickle.load(inp)
with open('/Users/josepsmachine/Documents/UNI/DL/dlnn-project_ia-group_10/dataset/val_dataset.pkl', 'rb') as inp:
    val_dataset = pickle.load(inp)

#create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
word2vec_embb = api.load('glove-wiki-gigaword-50')
vocabulary = word2vec_embb.index_to_key #save vocabulary list

word2vec_embb = torch.FloatTensor(word2vec_embb.vectors)
word2vec_embb = nn.Embedding.from_pretrained(word2vec_embb)
word2vec_embb.requires_grad_ = False #freeze word2vec embeddding layer

#load universal sentence encoder to define our own loss function
sntc_enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#define cross entropy loss function
cross_entrop = nn.CrossEntropyLoss()

def train(config: Dict = None):
    with wandb.init(config, project="dl2023_imagecaptioning", entity = "dl2023team"):
        config = wandb.config
        config = nested_dict(config)
        
        #define model for this run
        with open('aux.pkl', 'rb') as f:
            config = pickle.load(f)
        if config["embedding_layer"] == "word2vec":
            emb_layer = word2vec_embb
        else:
            #the learnt embedding layer will have the same vocab size as word2vec for comparaison reasons.
            emb_layer = nn.Embedding(num_embeddings=word2vec_embb.weight.shape[0], embedding_dim=config["hidden_size"])

        model = LSTMModel(input_dim=512,embedding_layer=emb_layer,hidden_dim=config["hidden_size"],n_layers=config['num_layers'])
        model.init_weights()
        model.to(device)

        #define optimizer for this run
        optimizer_config = config["optimizer"]
        if optimizer_config["type"] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = optimizer_config['lr'])
        
        #define loss for this run
        if config["loss_funct"] == "crossentropy":
            def loss_funct(pred,ref): 
                ref = ref.long()
                loss = cross_entrop(pred,ref)
                return loss
        else:
            def loss_funct(pred,ref):
                ref = ref.long()
                loss1 = cross_entrop(pred,ref)

                pred_keys = torch.argmax(pred,axis=-1)
                
                pred_sntc = ""
                target_sntc = ""
                for batch_ref,batch_pred in zip(ref.int(),pred_keys):
                    pred_sntc += " " + vocabulary[batch_pred.item()]
                    target_sntc += " " + vocabulary[batch_ref.item()]

                #ic(caption)
                #ic(target_sntc)
                #ic(pred_sntc)

                embeddings = sntc_enc.encode([pred_sntc, target_sntc])
                similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))
                loss = similarity[0][0]*COSINE_SIM_IMPORTANCE + loss1*(1-COSINE_SIM_IMPORTANCE)
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
                X1 = X1.to(device) 
                X2 = X2.to(device)
                out,h,c = model(X1,X2)
                #pred_keys = torch.argmax(out,axis=-1)
                ref = X2.float()
                #change ref to join batch and sequence dim
                ref = ref.view(ref.shape[0]*ref.shape[1])

                #same for the out logits
                #change ref to join batch and sequence dim
                out = out.view(out.shape[0]*out.shape[1],out.shape[2])
                
                loss = loss_funct(out,ref)
                loss.backward()
                optimizer.step()    
                training_losses.append(loss.item())
                #progress_bar.set_postfix({'Batch Loss': loss.item()})

            average_training_loss = sum(training_losses) / len(training_losses) # renamed from avg_loss
            #average_training_loss = np.power(dataset.denormalize_values(np.sqrt(average_training_loss),scaler),2)
            wandb.log({'Train_Epoch_Loss': average_training_loss})

            model.eval()  
            with torch.no_grad():  
                validation_losses = [] # renamed from val_losses
                for X1,X2,caption in tqdm(val_dataloader, desc='Validation'):
                    X1 = X1.to(device) 
                    X2 = X2.to(device)

                    out,h,c = model(X1,X2)

                    ref = ref.view(ref.shape[0]*ref.shape[1])
                    out = out.view(out.shape[0]*out.shape[1],out.shape[2])
                    
                    loss = loss_funct(out,ref)

                    validation_losses.append(loss.item())

                average_validation_loss = sum(validation_losses) / len(validation_losses) # renamed from avg_val_loss
                #average_validation_loss = np.power(dataset.denormalize_values(np.sqrt(average_validation_loss),scaler),2)
                wandb.log({'Validation_Epoch_Loss': average_validation_loss})

            if average_training_loss < best_loss:
                best_loss = average_training_loss
                torch.save(model.state_dict(), 'models/LSTM&resnet18/LSTM&resnet18.pt')
                wandb.save('LSTM&resnet18.pt')
                print(f"Model saved at {'LSTM&resnet18.pt'}")

        wandb.finish()


#run the agent
wandb.agent(sweep_id, function=train)