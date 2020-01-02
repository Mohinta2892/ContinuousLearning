from models.binary.SimpleNN import SimpleNN
from models.binary.utilities import *
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
    model_EWC.load_state_dict(torch.load(MODEL_EWC_PATH))

    model_AND.eval()
    model_OR.eval()
    model_EWC.eval()
else:    
    print("Models does not exist. Creating models.")
    
    train_model(model_AND, 'data/binary/data_AND.csv', device)
    train_model(model_OR, 'data/binary/data_OR.csv', device)
    train_ewc(model_EWC, 10000, device, ['data/binary/data_AND.csv', 'data/binary/data_OR.csv'])

    print("Saving model.")

    torch.save(model_AND.state_dict(), MODEL_AND_PATH)
    torch.save(model_OR.state_dict(), MODEL_OR_PATH)
    torch.save(model_EWC.state_dict(), MODEL_EWC_PATH)

# Get the data
dataset = BinaryData(csv_path='data/binary/data_AND.csv')
dataloaderAND = DataLoader(dataset, batch_size=4, shuffle=True)
dataset = BinaryData(csv_path='data/binary/data_OR.csv')
dataloaderOR = DataLoader(dataset, batch_size=4, shuffle=True)

print(f'AND Model Accuracy: {test_model(model_AND, dataloaderAND) * 100}%')
print(f'OR Model Accuracy: {test_model(model_OR, dataloaderOR) * 100}%')
print(f'EWC Model Accuracy for AND: {test_model(model_EWC, dataloaderAND) * 100}%')
print(f'EWC Model Accuracy for OR: {test_model(model_EWC, dataloaderOR) * 100}%')

inputs = torch.tensor([[0,0],[0,1],[1,0],[1,1]])
predAnd = check_model(torch.cat((inputs, torch.zeros([4,1]).long()), dim=1), model_EWC)
predOr = check_model(torch.cat((inputs, torch.ones([4,1]).long()), dim=1), model_EWC)