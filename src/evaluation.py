import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import accuracy, nll, ece, get_optimal_temperature


def evaluate_base(model, test_loader, valid_loader, device, inference=True):
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        
        final_acc = 0.0
        final_nll = 0.0
        final_ece = 0.0
        final_cnll = 0.0
        final_cece = 0.0
        
        for valid_output, valid_labels in valid_loader:
            valid_output, valid_labels = valid_output.to(device), valid_labels.to(device)
            valid_output = model(valid_output)
            t_opt = get_optimal_temperature(valid_output, valid_labels)
                
        for step, (images, labels) in tqdm(enumerate(test_loader)):
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            
            output = output.cpu()
            labels = labels.cpu()
            
            acc_ = accuracy(output, labels)
            nll_ = nll(output, labels)
            ece_ = ece(output, labels)
            cnll_ = nll(output, labels, t_opt=t_opt)
            cece_ = ece(output, labels, t_opt=t_opt)    
            
            final_acc += acc_
            final_nll += nll_
            final_ece += ece_
            final_cnll += cnll_
            final_cece += cece_
        
        final_acc = final_acc / len(test_loader)
        final_nll = final_nll / len(test_loader)
        final_ece = final_ece / len(test_loader)
        final_cnll = final_cnll / len(test_loader)
        final_cece = final_cece / len(test_loader)
        
        print(float(final_acc))
        print(float(final_nll))
        print(float(final_ece))
        print(float(final_cnll))
        print(float(final_cece)) 
                
    
def evaluate_ensemble(model, test_loader, valid_loader, ensemble_size, device):
    """about try, except:
        try -> to evaluate teacher ensemble model
        except -> to evaluate batch ensemble model
    """
    try:
        umodel = []
        for tmodel in model:
            tmodel.to(device)
            tmodel.eval()
            umodel.append(tmodel)
    except:
        model.to(device)
        model.eval()
    
    with torch.no_grad():
        
        final_acc = 0.0
        final_nll = 0.0
        final_ece = 0.0
        final_cnll = 0.0
        final_cece = 0.0
        
        try:
            for images, valid_labels in valid_loader:
                images, valid_labels = images.to(device), valid_labels.to(device)
                valid_output = []
                for tmodel in umodel:
                    valid_output.append(tmodel(images))
                valid_output = torch.stack(valid_output, dim=-1)
                valid_output = torch.mean(valid_output, dim=-1)
                t_opt = get_optimal_temperature(valid_output, valid_labels)
        
        except:
            for valid_output, valid_labels in valid_loader:
                valid_output, valid_labels = valid_output.to(device), valid_labels.to(device)
                valid_output = model(valid_output)
                t_opt = get_optimal_temperature(valid_output, valid_labels)
                
        for step, (images, labels) in tqdm(enumerate(test_loader)):
            
            images, labels = images.to(device), labels.to(device)
            
            try:
                output = []
                for tmodel in umodel:
                    output.append(tmodel(images))
                output = torch.stack(output, dim=1)
                output = torch.mean(output, dim=1)   
                
            except: 
                images = images.expand(ensemble_size,  -1, -1, -1, -1)
                images = images.reshape(-1, images.size(-3), images.size(-2), images.size(-1))
                output = model(images)
                _, s_out = output.size()
                output = output.reshape(ensemble_size, -1, s_out)
                output = torch.mean(output, dim=0)
                
            acc_ = accuracy(output, labels)
            nll_ = nll(output, labels)
            ece_ = ece(output, labels) 
            
            cnll_ = nll(output, labels, t_opt=t_opt)
            cece_ = ece(output, labels, t_opt=t_opt)   
             
            final_acc += acc_
            final_nll += nll_
            final_ece += ece_
            final_cnll += cnll_
            final_cece += cece_
        
        final_acc = final_acc / len(test_loader)
        final_nll = final_nll / len(test_loader)
        final_ece = final_ece / len(test_loader)
        final_cnll = final_cnll / len(test_loader)
        final_cece = final_cece / len(test_loader)

        print(float(final_acc))
        print(float(final_nll))
        print(float(final_ece))
        print(float(final_cnll))
        print(float(final_cece)) 