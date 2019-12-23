import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from BinaryData import BinaryData
from EwcNN import EWC

from tqdm import tqdm

def check_model(inputs, model):
    out = model(inputs)
    # We use a sigmoid function as we want the output to be between 0 and 1
    preds = torch.sigmoid(out).round()
    return preds.detach().numpy()

def test_model(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            X, y = data
            X = X.long()
            y = y.float()

            output = model(X.view(-1, 3))
            # print(output)
            for idx, out in enumerate(output):
                #print(torch.argmax(i), y[idx])
                if torch.sigmoid(out).round() == y[idx]:
                    correct += 1
                total += 1

    # Returns accuracy
    return round(correct/total, 3)

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

    # Train over the same data with different permutations
    for epoch in tqdm(range(EPOCS)):
        loss = normal_train(model, opt, loss_func, dataloader, device)
        if epoch % 200 == 0: print(loss.item())

    # Test the model and print the accuracy
    acc = test_model(model, dataloader)
    print("Accuracy: ", acc)

def normal_train(model: nn.Module, opt: torch.optim, loss_func:torch.nn, data_loader: torch.utils.data.DataLoader, device):
    epoch_loss = 0

    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device).long()
        labels = labels.to(device).float()

        opt.zero_grad()

        output = model(inputs)
        loss = loss_func(output.view(-1), labels)
        epoch_loss += loss.item()
        loss.backward()
        opt.step()

    # return epoch_loss / len(data_loader)
    return loss


# model, data_path, device
def train_ewc(model, importance, device, data_paths):

    print("########################################################")
    print("###                  Training model                  ###")
    print("########################################################")

    # Get the data
    dataloader = []
    for i, path in enumerate(data_paths):
        dataset = BinaryData(csv_path=path)
        dataloader.append(DataLoader(dataset, batch_size=4, shuffle=True))

    EPOCS = 1000
    sample_size = 100

    opt = optim.SGD(params=model.parameters(), lr=0.1)
    loss_func = nn.BCEWithLogitsLoss()
    loss, acc, ewc = {}, {}, {}

    # Two tasks: AND / OR
    for task in range(2):
        loss[task] = []
        acc[task] = []

        if task == 0:
            a = 1
            for epoch in tqdm(range(EPOCS)):
                loss[task].append(normal_train(model, opt, loss_func, dataloader[task], device))
                acc[task].append(test_model(model, dataloader[task]))
                if(epoch % 200 == 0): print(loss[task][-1])

        else:
            old_tasks = []
            for sub_task in range(task):
                old_tasks.append(dataloader[sub_task].dataset.get_sample())

            # Train the model EWC
            ewc = EWC(model, old_tasks)
            for epoch in tqdm(range(EPOCS)):

                # Get the loss
                ls = ewc_train(model, opt, loss_func, dataloader[task], ewc, importance, device)
                loss[task].append(ls)

                # Get the accuraccy + the accuracy of the old tasks
                for sub_task in range(task + 1):
                    acc[sub_task].append(test_model(model, dataloader[sub_task]))

                if(epoch % 200 == 0): print(loss[task][-1])


    # Test the model and print the accuracy
    print(f'AND Accuracy: {test_model(model, dataloader[0]) * 100}%')
    print(f'OR Accuracy: {test_model(model, dataloader[1]) * 100}%')

    return loss, acc


def ewc_train(model: nn.Module, opt: torch.optim, loss_func:torch.nn, data_loader: torch.utils.data.DataLoader, ewc: EWC, importance: float, device):
    epoch_loss = 0

    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device).long()
        labels = labels.to(device).float()

        opt.zero_grad()

        output = model(inputs)
        loss = loss_func(output.view(-1), labels) + importance * ewc.penalty(model)
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

    # return epoch_loss / len(data_loader)
    return loss