from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from data.MnistData import getSamples
from models.EwcNN import EWC

def train_and_test(model, trainset, testset, device, task, oldSets, optimizer, EPOCHS, importance=3000):
    loss, acc = train_ewc(model, trainset, testset, device, task, oldSets, optimizer, EPOCHS=EPOCHS, importance=importance)

    for t in range(task):
        accuracy = test(model, testset[t], device)
        print(f'Accuracy for task {t+1}: {accuracy}')

    return loss, acc

def getDevice(forceCPU = True):
    if not forceCPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    return device

def test(model, testset, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testset:
            X, y = data
            X, y = X.to(device), y.to(device)
            output = model(X.view(-1,784))
            
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

            # print(torch.argmax(model(X[0].view(-1,784))[0]))
            # plt.imshow(X[0].view(28,28))
            # plt.show()

    # Return the accuracy
    return round(correct/total, 3)

def train(model, trainset, device, EPOCHS=3):
    print("########################################################")
    print("###                  Training model                  ###")
    print("########################################################")

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(EPOCHS)): 
        loss = _normal_train(model, trainset, device, loss_function, optimizer)
        print(loss)

    return loss


def train_ewc(model, trainset, testset, device, task, oldSets, optimizer, EPOCHS=5, importance=3000):
    print("########################################################")
    print(f'###          Training model for task {task}               ###')
    print("########################################################")

    sample_size = 180
    loss_function = nn.CrossEntropyLoss()

    loss = []
    accuracy = []

    for idx in range(task):
        accuracy.append([])

    if task == 1:
        for epoch in tqdm(range(EPOCHS)):
            l = _normal_train(model, trainset, device, loss_function, optimizer)
            acc = test(model, testset[0], device)

            loss.append(l)
            accuracy[0].append(acc)
            print(l)

    else:
        old_tasks = []
        for t in range(task-1):
            old_tasks = old_tasks + getSamples(oldSets[t], sample_size)

        ewc=EWC(model, old_tasks)
        for epoch in tqdm(range(EPOCHS)):

            # if(epoch % 5 == 0):
            #     ewc=EWC(model, old_tasks)

            l = _ewc_train(model, optimizer, loss_function, trainset, EWC(model, old_tasks), importance, device)
            loss.append(l)

            for i, testSet in enumerate(testset):
                acc = test(model, testSet, device)
                accuracy[i].append(acc)
                print(f'Accuracy for task {i}: {acc}')

            print(loss[epoch])

    return loss, accuracy


def _normal_train(model, trainset, device, loss_function, optimizer):
    for data in trainset:         
        X, y = data               
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad() 
        output = model(X.view(-1,784)) 

        loss = loss_function(output, y)
        loss.backward() 
        optimizer.step()

    return loss.item()

def _ewc_train(model, optimizer, loss_function, trainset, ewc, importance, device):
    for data in trainset:
        X, y = data               
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(X.view(-1, 784))

        loss = loss_function(output, y) + importance * ewc.penalty(model)
        # if(i % 100 == 0): print(loss)
        loss.backward()
        optimizer.step()

    return loss.item()
