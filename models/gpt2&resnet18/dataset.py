import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
from transformers import GPT2Tokenizer
from tqdm import tqdm

class image_caption_dataset(Dataset):
    def __init__(self,images_pth,captions_txt_pth,n_im=0):
        self.df = pd.read_csv(captions_txt_pth)
        if n_im > 0:
            self.df = self.df[:n_im]
        
        self.im_pth = images_pth
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        #pipeline of transforms we might want to do
        self.transform = transforms.Compose([ #image net dataset normalisations.
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
        ])
        
        #load all images to RAM so we can access them faster
        self.images = []
        self.tokenized = []
        for i, row in  tqdm(self.df.iterrows(),total=len(self.df)):
            with Image.open(self.im_pth + row["image"]) as image: #context manager to open and close properly
                self.images.append(self.transform(transforms.functional.pil_to_tensor(image).float()))
                #the tokenized text is going to have a <s> token at the start and a <|endoftext|> at the end
                self.df.iloc[i]["caption"] = "<s>"+row["caption"]+"<|endoftext|>"
                self.tokenized.append(self.gpt2_tokenizer("<s>"+row["caption"]+"<|endoftext|>",return_tensors="pt")["input_ids"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Different operations and preprocessing can be done to the image previously with the following parameters
        self.
        Different return modes:

        Mode 0:
        - y is x with

        Mode 1:
        - y is encoded as a list of vectors from the pretrained embedding (????)

        Mode 2:
        - y is encoded using the pretrained model (????) for sentence embedding
        """
        #get row by the given index
        row = self.df.iloc[index]
        
        #load tag and image for the sample
        tag = row["caption"]
        #image
        im = self.images[index]
        tok = self.tokenized[index]
        
        #apply the transformations we want 
        return {"x":im,"y":tag,"y_tokenized":tok}