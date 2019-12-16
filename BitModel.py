from BinaryData import BinaryData
from SimpleNN import SimpleNN
import numpy as np
import os.path

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_model(model, data_path):
    print("########################################################")
    print("###                  Training model                  ###")
    print("########################################################")

    # Get the data
    dataset = BinaryData(csv_path=data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    opt = optim.SGD(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    EPOCS = 10000

    for epoch in tqdm(range(EPOCS)):
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()

            opt.zero_grad()

            output = model(inputs)
            loss = loss_func(output, labels)
            loss.backward()
            opt.step()

        if epoch % 2000 == 0: print(loss)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            X, y = data
            output = model(X.view(-1, 2))
            # print(output)
            for idx, i in enumerate(output):
                #print(torch.argmax(i), y[idx])
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))

MODEL_AND_PATH = 'models/binary/model_AND.pth'
MODEL_OR_PATH = 'models/binary/model_OR.pth'

device = torch.device("cpu")
model_AND = SimpleNN(inputSize=2, hiddenLayer=10, outputSize=2).to(device)
model_OR = SimpleNN(inputSize=2, hiddenLayer=10, outputSize=2).to(device)

if(os.path.isfile(MODEL_AND_PATH) and os.path.isfile(MODEL_OR_PATH)):
    print("Models exists. Using old models.")
    model_AND.load_state_dict(torch.load(MODEL_AND_PATH))
    model_OR.load_state_dict(torch.load(MODEL_OR_PATH))

    model_AND.eval()
    model_OR.eval()
else:    
    print("Models does not exist. Creating models.")
    
    train_model(model_AND, 'data/binary/data_AND.csv')
    train_model(model_OR, 'data/binary/data_OR.csv')

    print("Saving model.")
    torch.save(model_AND.state_dict(), MODEL_AND_PATH)
    torch.save(model_OR.state_dict(), MODEL_OR_PATH)

prediction_and = model_AND(torch.tensor([[1,0],[1,1],[0,0],[0,1]]))
prediction_and = prediction_and.argmax(dim=1)

prediction_or = model_OR(torch.tensor([[1,0],[1,1],[0,0],[0,1]]))
prediction_or = prediction_or.argmax(dim=1)

print("STOP")