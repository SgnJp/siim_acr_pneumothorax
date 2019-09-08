import torch
import torch.nn as nn
import numpy as np
import tqdm

def train(model, 
          train_loader, 
          optimizer, 
          loss, 
          metrics_callback = None, 
          scheduler = None, 
          gradient_accumulation = 1,
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
          
    model.train()

    optimizer.zero_grad()
    for batch_idx, datas in enumerate(tqdm.tqdm(train_loader), 1):
        inputs, targets = datas
        targets = targets.type(torch.FloatTensor)
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        
        
        loss_output = loss(outputs, targets)
        loss_output.backward()

        if batch_idx % gradient_accumulation == 0 and batch_idx > 0:
            optimizer.step()
            optimizer.zero_grad()

        if not scheduler is None:
            scheduler.step()
            
        if not metrics_callback is None:
            metrics_callback.update(outputs, targets)

    ### If there is still something left
    if batch_idx % gradient_accumulation != 0:
        optimizer.step()
        optimizer.zero_grad()


def test(model, val_loader, metrics_callback, 
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    
    with torch.no_grad():
        for batch_idx, datas in enumerate(val_loader):
            inputs, targets = datas
            targets = targets.type(torch.FloatTensor)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            metrics_callback.update(outputs, targets)


def predict(model, loader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    result = []
    model.eval()
    
    with torch.no_grad():
        for datas in tqdm.tqdm(loader):
            ## TODO: Make it work without targets
            inputs = datas[0]
            inputs = inputs.to(device)

            outputs = model(inputs)
            result.append(outputs.cpu().numpy())

    return np.concatenate(result, axis=0)
