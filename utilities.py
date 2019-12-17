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

def test_model(model):
    inputs = torch.tensor([[0,0],[0,1],[1,0],[1,1]])
    pred = check_model(inputs, model)

def train_model(model, data_path, device):
    print("########################################################")
    print("###                  Training model                  ###")
    print("########################################################")

    # Get the data
    dataset = BinaryData(csv_path=data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    opt = optim.SGD(model.parameters(), lr=0.001)
    loss_func = nn.BCEWithLogitsLoss()

    EPOCS = 10000

    for epoch in tqdm(range(EPOCS)):
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device).long()
            labels = labels.to(device).float()

            opt.zero_grad()

            output = model(inputs)
            loss = loss_func(output.view(-1), labels)
            loss.backward()
            opt.step()
        if epoch % 2000 == 0: print(loss.item())

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            X, y = data
            X = X.to(device).long()
            y = y.to(device).float()

            output = model(X.view(-1, 3))
            # print(output)
            for idx, out in enumerate(output):
                #print(torch.argmax(i), y[idx])
                if torch.sigmoid(out).round() == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))

def train_ewc(epochs, importance, device, dataloader):

    # Create model
    # model = MLP(hidden_size)

    # opt = optim.SGD(params=model.parameters(), lr=1e-3)
    # loss_func = nn.BCEWithLogitsLoss()

    # loss, acc, ewc = {}, {}, {}

    # # Two tasks: AND / OR
    # for task in range(2):
    #     loss[task] = []
    #     acc[task] = []

    #     if task == 0:
    #         for _ in tqdm(range(epochs)):
    #             epoch_loss = 0
                
    #             for i, (inputs, labels) in enumerate(dataloader[task]):
    #                 inputs = inputs.to(device).long()
    #                 labels = labels.to(device).float()

    #                 opt.zero_grad()

    #                 output = model(inputs)
    #                 loss = loss_func(output.view(-1), labels)
    #                 loss.backward()
    #                 epoch_loss += loss.item()
    #                 opt.step()

    #             loss[task].append(epoch_loss / len(dataloader[task]))
               
    #             acc[task].append(test(model, test_loader[task]))
    #     else:
    #         old_tasks = []
    #         for sub_task in range(task):
    #             old_tasks = old_tasks + train_loader[sub_task].dataset.get_sample(sample_size)
    #         old_tasks = random.sample(old_tasks, k=sample_size)
    #         for _ in tqdm(range(epochs)):
    #             loss[task].append(ewc_train(model, optimizer, train_loader[task], EWC(model, old_tasks), importance))
    #             for sub_task in range(task + 1):
    #                 acc[sub_task].append(test(model, test_loader[sub_task]))

    # return loss, acc
    return 0
