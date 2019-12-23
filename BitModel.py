from SimpleNN import SimpleNN
from utilities import *
import os.path
import torch

RETRAIN_MODEL = True

MODEL_AND_PATH = 'models/binary/model_AND.pth'
MODEL_OR_PATH = 'models/binary/model_OR.pth'
MODEL_EWC_PATH = 'models/binary/model_EWC.pth'

device = torch.device("cpu")

model_AND = SimpleNN(inputSize=3, hiddenLayer=10, outputSize=1).to(device)
model_OR = SimpleNN(inputSize=3, hiddenLayer=10, outputSize=1).to(device)
model_EWC = SimpleNN(inputSize=3, hiddenLayer=10, outputSize=1).to(device)

if(not(RETRAIN_MODEL) and os.path.isfile(MODEL_AND_PATH) and os.path.isfile(MODEL_OR_PATH)):
    print("Models exists. Using old models.")
    model_AND.load_state_dict(torch.load(MODEL_AND_PATH))
    model_OR.load_state_dict(torch.load(MODEL_OR_PATH))

    model_AND.eval()
    model_OR.eval()
else:    
    print("Models does not exist. Creating models.")
    
    # train_model(model_AND, 'data/binary/data_AND.csv', device)
    # train_model(model_OR, 'data/binary/data_OR.csv', device)
    loss, acc = train_ewc(model_EWC, 10000, device, ['data/binary/data_AND.csv', 'data/binary/data_OR.csv'])


    print("Saving model.")
    # torch.save(model_AND.state_dict(), MODEL_AND_PATH)
    # torch.save(model_OR.state_dict(), MODEL_OR_PATH)


inputs = torch.tensor([[0,0],[0,1],[1,0],[1,1]])
predAnd = check_model(torch.cat((inputs, torch.zeros([4,1]).long()), dim=1), model_EWC)
predOr = check_model(torch.cat((inputs, torch.ones([4,1]).long()), dim=1), model_EWC)

print(predAnd)
print(predOr)