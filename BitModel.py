from SimpleNN import SimpleNN
from utilities import *
import os.path
import torch

RETRAIN_MODEL = False

MODEL_AND_PATH = 'models/binary/model_AND.pth'
MODEL_OR_PATH = 'models/binary/model_OR.pth'

device = torch.device("cpu")

model_AND = SimpleNN(inputSize=2, hiddenLayer=3, outputSize=1).to(device)
model_OR = SimpleNN(inputSize=2, hiddenLayer=3, outputSize=1).to(device)

if(not(RETRAIN_MODEL) and os.path.isfile(MODEL_AND_PATH) and os.path.isfile(MODEL_OR_PATH)):
    print("Models exists. Using old models.")
    model_AND.load_state_dict(torch.load(MODEL_AND_PATH))
    model_OR.load_state_dict(torch.load(MODEL_OR_PATH))

    model_AND.eval()
    model_OR.eval()
else:    
    print("Models does not exist. Creating models.")
    
    train_model(model_AND, 'data/binary/data_AND.csv', device)
    train_model(model_OR, 'data/binary/data_OR.csv', device)

    print("Saving model.")
    torch.save(model_AND.state_dict(), MODEL_AND_PATH)
    torch.save(model_OR.state_dict(), MODEL_OR_PATH)


inputs = torch.tensor([[0,0],[0,1],[1,0],[1,1]])
predAnd = check_model(inputs, model_AND)
predOr = check_model(inputs, model_OR)

print(predAnd)
print(predOr)