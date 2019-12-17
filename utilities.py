import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from BinaryData import BinaryData

from tqdm import tqdm

def check_model(inputs, model):
    out = model(inputs)
    # We use a sigmoid function as we want the output to be between 0 and 1
    preds = torch.sigmoid(out).round()
    return preds.detach().numpy()


def train_model(model, data_path, device):
    print("########################################################")
    print("###                  Training model                  ###")
    print("########################################################")

    # Get the data
    dataset = BinaryData(csv_path=data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    opt = optim.SGD(model.parameters(), lr=0.1)
    loss_func = nn.BCEWithLogitsLoss()

    EPOCS = 1000

    for epoch in tqdm(range(EPOCS)):
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device).long()
            labels = labels.to(device).float()

            opt.zero_grad()

            output = model(inputs)
            loss = loss_func(output.view(-1), labels)
            loss.backward()
            opt.step()
        if epoch % 200 == 0: print(loss.item())

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            X, y = data
            X = X.to(device).long()
            y = y.to(device).float()

            output = model(X.view(-1, 2))
            # print(output)
            for idx, out in enumerate(output):
                #print(torch.argmax(i), y[idx])
                if torch.sigmoid(out).round() == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))