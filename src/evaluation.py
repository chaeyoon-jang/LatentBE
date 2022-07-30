import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from utils import accuracy, nll, ece, temperature_scaling

def get_optimal_temperature(confidences, labels):
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
            cnll_ = nll(temperature_scaling(output, t_opt, log_input=False), labels)
            cece_ = ece(temperature_scaling(output, t_opt, log_input=False), labels)    
            
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
        
        print(final_acc)
        print(final_nll)
        print(final_ece)
        print(final_cnll)
        print(final_cece) 
                
        #print(f"acc : {final_acc : .3f} | nll : {final_nll : .3f} | ece : {final_ece : .3f}")
        #print(f"cnll : {final_cece : .3f} | cece : {final_cnll : .3f}")
    
def evaluate_latentbe(model, test_loader, device):
    
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
            
            acc_ = accuracy(output, labels)
            nll_ = nll(output, labels)
            ece_ = ece(output, labels)
            #t_opt = get_optimal_temperature(output, labels)
            cnll_ = nll(temperature_scaling(output, 5.0, log_input=False))
            cece_ = ece(temperature_scaling(output, 5.0, log_input=False))    
            
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
        
        print(f"acc : {final_acc : .3f} | nll : {final_nll : .3f} | ece : {final_ece : .3f}")
        print(f"cnll : {final_cece : .3f} | cece : {final_cnll : .3f}")