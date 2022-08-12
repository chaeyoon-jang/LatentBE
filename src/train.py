import os
import os.path as p
import wandb
import numpy as np
import torch
import torch.nn as nn

def save_ckpt(ckpt_path, model, epoch, train_loss, best_loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict" : model.state_dict(),
        "train_loss" : train_loss,
        "best_loss" : best_loss
    }, ckpt_path)

def validate(model, valid_loader, device):

    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    with torch.no_grad():
        epoch_loss = 0.0
        for step, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            loss = criterion(output, labels)

            epoch_loss += loss.item()

        valid_loss = epoch_loss / len(valid_loader)

    return valid_loss
    
def train(
        model = None,
        train_loader = None,
        valid_loader = None,
        n_epochs = None,
        device = None,
        learning_rate = None,
        logging_step = 300,
        seed = None,
        ckpt_path = './ckpt'
        ):
    
    losses = []
    train_loss = []
    best_loss = float('inf')
    
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
    #                                    lr_lambda= lambda epoch: 0.95 ** n_epochs,
    #                                    last_epoch=-1,
    #                                    verbose=False)
    
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            #print(images.size())
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            
            loss = criterion(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            wandb.log({
                "train_loss": epoch_loss / (step + 1),
            })

            #if (step + 1) % logging_step == 0:
            #    print(f"[Epoch {epoch + 1}/{n_epochs}] Step {step  + 1}/{len(train_loader)} | loss: {epoch_loss/(step + 1): .3f}")
                
        #scheduler.step()
        train_loss.append(epoch_loss / len(train_loader))
        valid_loss = validate(model, valid_loader, device)
        
        print(f"[Epoch {epoch + 1}/{n_epochs}] Step {step  + 1}/{len(train_loader)} | loss: {epoch_loss/(step + 1): .3f} | valid loss: {valid_loss: .3f}")

        if (epoch + 1) % 30 == 0 :
            if valid_loss < best_loss:
                best_loss = valid_loss
                save_ckpt(
                    ckpt_path=p.join(ckpt_path, f"cifar_{seed}_teacher_model_checkpoint_epoch_{epoch + 1}.pt"), 
                    model=model, epoch=epoch + 1, 
                    train_loss=train_loss, best_loss=valid_loss
                )
                print(f"Success to save checkpoint.")
            else:
                print("No improvement detected. Skipping save")
