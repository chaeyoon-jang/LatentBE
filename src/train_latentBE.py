import os
import os.path as p
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

def save_ckpt(ckpt_path, model, epoch, best_loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict" : model.state_dict(),
        "best_loss" : best_loss
    }, ckpt_path)

def validate(model, valid_loader, device):

    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    with torch.no_grad():
        epoch_loss = 0.0
        for step, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device)
            
            images = torch.cat((images, images, images, images), 0)

            output = model(images)
            output = output.reshape(4, int(output.size(-2)/4), output.size(-1))
            output = torch.mean(output, dim=0)
            loss = criterion(output, labels)

            epoch_loss += loss.item()

        valid_loss = epoch_loss / len(valid_loader)

    return valid_loss

def train_latentbe(
        tmodel,
        smodel,
        train_loader,
        valid_loader,
        n_epochs,
        device,
        learning_rate ,
        ckpt = "./ckpt2",
        temperature = 20.0,
        alpha = 0.9,
        weight_decay = 5e-4
        ):
    
    optimizer = torch.optim.Adam(smodel.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    kld_loss  = nn.KLDivLoss(reduction='batchmean')
    
    losses = []
    train_loss = []
    best_loss = float('inf')
    best_epoch = 0
    current_ckpt = ""
    
    for num, model in enumerate(tmodel):
        seed = 42 + num
        ckpt_p = p.join(ckpt, str(seed)+"_teacher_model_checkpoint.pt")
        model.load_state_dict(torch.load(ckpt_p)['model_state_dict'])
        model.to(device)
        model.eval()
        tmodel[num] = model
    
    smodel.to(device)
    for epoch in range(n_epochs):
        smodel.train()
        epoch_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            
            images, labels = images.to(device), labels.to(device)
            #sub_images = images.view(len(tmodel), -1, images.size(-3), images.size(-2), images.size(-1))
            
            teacher_predicts = []
            for sub_idx, model in enumerate(tmodel):
                teacher_predicts.append(model(images))
            
            images = images.expand(len(tmodel),  -1, -1, -1, -1)
            images = images.reshape(-1, images.size(-3), images.size(-2), images.size(-1))
            output = smodel(images)
            
            # loss_ce
            labels = labels.expand(len(tmodel), -1)
            labels = labels.reshape(1, -1)
            labels = torch.squeeze(labels)
            loss_ce = criterion(output, labels)
            
            # loss_kd
            teacher_predicts = torch.cat(teacher_predicts, 0)
            teacher_predicts = F.log_softmax(teacher_predicts/temperature, dim=-1)
            student_predicts = F.log_softmax(output/temperature, dim=-1)
            loss_kd = kld_loss(student_predicts, teacher_predicts) * (temperature **2)
            
            # loss_wd (L2 weight decay = Gaussian Prior)
            # loss_wd = sum(torch.linalg.norm(p, 2) for p in smodel.parameters())
            
            loss_wd = 0.5 * sum([torch.sum(p**2) for p in smodel.parameters()])
            
            params_r = [param for name, param in smodel.state_dict().items() if 'r_factor' in name]
            params_s = [param for name, param in smodel.state_dict().items() if 's_factor' in name]
            
            loss_wd -= 0.5 * sum([torch.sum(p**2) for p in params_r])
            loss_wd -= 0.5 * sum([torch.sum(p**2) for p in params_s])
            loss_wd += 0.5 * sum([torch.sum((1-p)**2) for p in params_r]) / len(tmodel)
            loss_wd += 0.5 * sum([torch.sum((1-p)**2) for p in params_s]) / len(tmodel)
            
            # loss_total                
            loss = (1-alpha) * loss_ce + alpha * loss_kd + weight_decay * loss_wd
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss_ce.item()
            
        train_loss.append(epoch_loss / len(train_loader))
        valid_loss = validate(smodel, valid_loader, device)
        
        wandb.log({
                "train_loss": epoch_loss / (step + 1),
                "valid_loss": valid_loss
            })
            
        print(f"[Epoch {epoch + 1}/{n_epochs}] Step {step  + 1}/{len(train_loader)} | loss: {epoch_loss/(step + 1): .3f} | valid loss: {valid_loss: .3f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch 
            current_ckpt = f"LatentBE_model_checkpoint_epoch_{epoch + 1}.pt"
            print(f"Success to save checkpoint.")
        else:
            print("No improvement detected. Skipping save")
            
    save_ckpt(
        ckpt_path=p.join(ckpt, current_ckpt), 
        model=smodel, epoch=best_epoch + 1, 
        best_loss=best_loss
    )