# %%
from models import caption_generator

# %%
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

# %%
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
resnet18 = models.resnet18(pretrained=True)

# %%
device = "mps"

# %%
model = caption_generator(gpt2,resnet18,tokens_per_img=50,batch_size=1)
model.to(device)

# %%
from dataset import image_caption_dataset

# %%
dataset = image_caption_dataset("/Users/josepsmachine/Documents/UNI/DL/dlnn-project_ia-group_10_OLDDD/dataset/Images/","/Users/josepsmachine/Documents/UNI/DL/dlnn-project_ia-group_10_OLDDD/dataset/captions.txt",n_im=100)


# %%
from torch.utils.data import DataLoader

# %%
dataloader = DataLoader(dataset, batch_size=1, shuffle=True) 

# %%
from archive.train import train

# %%
train(model,dataloader,epochs=50,lr=0.01)


