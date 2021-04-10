import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torchvision
from utils import to_img 


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def training(model,train_loader,criterion,optimizer):
    '''
    Training one epoch of the model 

    return: Training loss of one epoch
    '''
    model.train()
    avg_loss=[]

    for batch_num, (feats, labels) in enumerate(train_loader):
        feats, labels = feats.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(feats)
        loss=criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        avg_loss.append(loss.item())
            
        torch.cuda.empty_cache()
        del feats
        del labels
        del loss

    print('Train Loss: {:.6f}'.format(sum(avg_loss)/len(avg_loss)))
    return sum(avg_loss)/len(avg_loss)


def validation(model,test_loader,criterion):
    '''
    Validation for  one epoch of the model  

    return: Validation loss of one epoch
    '''
    model.eval()
    avg_loss=[]
    out=[]
    inp=[]
    
    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)

        outputs=model(feats)
        temp=outputs[0].detach().cpu().numpy()
        out.append(temp.reshape(-1, 80,640))
        temp2=feats[0].detach().cpu().numpy()
        # inp.append(temp2.reshape(-1, 180,360))
        loss=criterion(outputs,labels)
        avg_loss.append(loss.item())
        del feats
        del temp
        del temp2
        del labels
    model.train()
    print('Validation Loss: {:.4f}'.format(sum(avg_loss)/len(avg_loss)))

    return np.array(out)

def test(model, test_loader):
    '''
    Validation for  one epoch of the model  

    return: Validation loss of one epoch
    '''
    model.eval()
    out = []
    label = []
    
    for batch_num, (feats, labels) in enumerate(test_loader):
        feats = feats.to(device)

        outputs=model(feats)
        out.append(outputs[0, 0].detach().cpu().numpy()) ## Moudularize
        label.append(labels[0, 0].numpy())

        del feats
        del labels

    return np.array(label), np.array(out)








        












