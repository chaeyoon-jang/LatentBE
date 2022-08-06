import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ECELoss(nn.Module):
    "https://github.com/gpleiss/temperature_scaling"
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, t_opt=0):
        if t_opt != 0:
            softmaxes = F.softmax(logits/t_opt, dim=1)
        else:
            softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    
def accuracy(predicts, labels):
    score = torch.sum(torch.argmax(predicts, dim=1)==labels).item()
    score = score / labels.size(0)
    return score

def nll(outputs, targets, t_opt=0):
    if t_opt != 0:
        score = torch.nn.NLLLoss(reduction="sum")(F.log_softmax(outputs/t_opt, dim=-1), targets)    
    else: 
        score = torch.nn.NLLLoss(reduction="sum")(F.log_softmax(outputs, dim=-1), targets)
    return score.item()

def ece(predicts, labels, t_opt=0):
    metric = ECELoss(n_bins=15)
    score = metric(predicts, labels, t_opt)
    return score

def temperature_scaling(predicts, temperature, log_input=True):
    if log_input:
        return F.log_softmax(predicts / temperature, dim=-1)
    else:
        
        return F.softmax(torch.log(predicts) / temperature, dim=-1)