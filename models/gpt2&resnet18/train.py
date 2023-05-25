import torch.nn.functional as F
import torch.nn as nn
import torch

device = "mps"

def train(model,dataloader,epochs,lr):
    ref_loss = float("inf")
    entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch,data in enumerate(dataloader):
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            img = data["x"].to(device)
            caption = data["y_tokenized"].to(device).squeeze(0)
            caption_logits = model(img=img,train=True,caption=caption)
            loss = entropy_loss(F.softmax(caption_logits[0],dim=-1),caption[0])
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch} || batch:{batch} || loss:{loss}")
            if loss.item() < ref_loss:
                ref_loss = loss.item()
                torch.save(model.state_dict(), 'gpt2&resnet18.pt')
