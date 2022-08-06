import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from utils import accuracy, nll, ece, temperature_scaling

def get_optimal_temperature(confidences, labels):
    confidences = F.softmax(confidences, dim=-1)
    confidences = confidences.cpu()
    labels = labels.cpu()
    def obj(t):
        target = labels.cpu().numpy()
        return -np.log(
            1e-12 + np.exp(
                torch.log_softmax(
                    torch.log(
                        1e-12 + confidences
                    ) / t, dim=1
                ).data.numpy()
            )[np.arange(len(target)), target]
        ).mean()

    optimal_temperature = minimize(
        obj, 1.0, method="nelder-mead", options={"xtol": 1e-3}
    ).x[0]
    return optimal_temperature

def evaluate_base(model, test_loader, device, inference=True):
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        
        final_acc = 0.0
        final_nll = 0.0
        final_ece = 0.0
        final_cnll = 0.0
        final_cece = 0.0
        
        for step, (images, labels) in tqdm(enumerate(test_loader)):
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            
            output = output.cpu()
            labels = labels.cpu()
            
            acc_ = accuracy(output, labels)
            nll_ = nll(output, labels)
            ece_ = ece(output, labels)

            t_opt = get_optimal_temperature(output, labels)
            cnll_ = nll(output, labels, t_opt=2.0)
            cece_ = ece(output, labels, t_opt=2.0)    
            
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
        print(float(final_nll)/100)
        print(float(final_ece))
        print(float(final_cnll)/100)
        print(float(final_cece)) 
                
        #print(f"acc : {final_acc : .3f} | nll : {final_nll : .3f} | ece : {final_ece : .3f}")
        #print(f"cnll : {final_cece : .3f} | cece : {final_cnll : .3f}")
    
def evaluate_ensemble(model, test_loader, ensemble_size, device):
    
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
            cnll_ = nll(output, labels, t_opt=2.0)
            cece_ = ece(output, labels, t_opt=2.0)   
             
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
        print(float(final_nll)/100)
        print(float(final_ece))
        print(float(final_cnll)/100)
        print(float(final_cece)) 