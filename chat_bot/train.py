import json
from nltk_utils import *
import numpy as np
from model import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json','r') as file:
    intents=json.load(file)

tags=[]
all_words=[]
xy=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        token_pattn=tokenize(pattern)
        all_words.extend(token_pattn)
        xy.append((token_pattn,tag))

words_to_ignore=['!','?','.',',']
all_words=[stem(word) for word in all_words if word not in words_to_ignore]
all_words=sorted(set(all_words))
tags=sorted(set(tags))
# print(tags)

X_train,y_train=[],[]
for (pattern_sentence,tag) in xy:
    bag=bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    label=tags.index(tag)
    y_train.append(label)

X_train=np.array(X_train)
y_train=np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples
    
batch_size_t=8
input_size=len(all_words)
hidden_size=8
output_size=len(tags)
lr=0.001
num_epochs=1000
dataset=ChatDataset()
train_loader=DataLoader(dataset=dataset,batch_size=batch_size_t,shuffle=True,num_workers=0)

model=ChatNN(input_size,hidden_size,output_size)

loss_criteria=nn.CrossEntropyLoss()
optimiser=torch.optim.Adam(model.parameters(),lr=lr)

for epoch in range(num_epochs):
    for (words,label) in train_loader:
        ouputs=model(words)
        loss=loss_criteria(ouputs,label)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    if(epoch+1)%100==0:
        print(f'Epoch: {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'Final loss={loss.item():.4f}')

data={
    "model_dict":model.state_dict(),
    "input_size":input_size,
    "hidden_size":hidden_size,
    "output_size":output_size,
    "all_words":all_words,
    "tags":tags
}

FILE="model.pth"
torch.save(data,FILE)

print(f'Training completed and file saved successfully as {FILE}')
