import torch.nn.functional as F
import torch.nn as nn
import torch

device = "mps"

def train(model,dataloader,epochs,lr):
    entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch,data in enumerate(dataloader):
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            img = data["x"].to(device)
            caption = data["y_tokenized"].to(device).squeeze(0)
            caption_logits = model(img=img,train=True,caption=caption)
            loss = entropy_loss(F.softmax(caption_logits[0]),caption[0])
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch} || batch:{batch} || loss:{loss}")
