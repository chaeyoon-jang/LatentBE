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
            
            images = torch.cat((images, images, images, images), 0)

            output = model(images)
            output = output.reshape(4, int(output.size(-2)/4), output.size(-1))
            output = torch.mean(output, dim=0)
            loss = criterion(output, labels)

            epoch_loss += loss.item()

        valid_loss = epoch_loss / len(valid_loader)

    return valid_loss

def train_latentbe_div(
        tmodel,
        smodel,
        train_loader,
        valid_loader,
        n_epochs,
        device,
        learning_rate ,
        ckpt = "./ckpt2",
        temperature = 20.0,
        alpha = 0.9
        ):
    
    def loss_perturb(images):
        
        teacher_predicts = []
        for model in tmodel:
            teacher_predicts.append(model(images))
        select = list(torch.randint(len(tmodel), (2,)))
        teacher_predicts = [teacher_predicts[int(select[0])], teacher_predicts[int(select[1])]]

        images = torch.cat((images, images, images, images), 0)
        student_predicts = smodel(images)
        student_predicts = student_predicts.reshape(len(tmodel), int(student_predicts.size(0) // len(tmodel)), student_predicts.size(-1))
        student_predicts = [student_predicts[int(select[0]), : , :], student_predicts[int(select[1]), : , :]]
        
        sdiv = torch.mean(
            -torch.sum(
                student_predicts[0] * torch.exp(student_predicts[1]).detach(), axis=-1
                )
        )
        
        tdiv = torch.mean(
                -torch.sum(
                    teacher_predicts[0] * torch.exp(teacher_predicts[1]).detach(), axis=-1
                )
        )
        l = tdiv - sdiv
        return l, teacher_predicts

    optimizer = torch.optim.Adam(smodel.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    log_softmax = nn.LogSoftmax(dim=-1)
    
    losses = []
    train_loss = []
    best_loss = float('inf')
    
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

            loss_pe, teacher_predicts  = loss_perturb(images)
            
            loss_pe.backward()
            print(images.grad)
            
            grad_perturb = grad_perturb / torch.linalg.norm(
                grad_perturb.reshape(grad_perturb.shape[0], -1), axis=-1
            ).reshape(-1, 1, 1, 1) * (1.0 / 255.0) * (
                grad_perturb.reshape(grad_perturb.shape[0], -1).shape[1] ** 0.5
            )

            images = torch.clip(
                images + grad_perturb,
                a_min=0.0, a_max=1.0,
            )
                        
            images = torch.cat((images, images, images, images), 0)
            output = smodel(images)
            
            labels = torch.cat((labels, labels, labels, labels), 0)
            loss_ce = criterion(output, labels)
            
            teacher_predicts = torch.cat((teacher_predicts[0], teacher_predicts[1], teacher_predicts[2], teacher_predicts[3]), 0)
            teacher_predicts = log_softmax(teacher_predicts/temperature)
            student_predicts = log_softmax(output/temperature)
            loss_kd = torch.mean(-torch.sum(student_predicts* torch.exp(teacher_predicts), axis=-1)) * (temperature**2)
            
            loss = (1-alpha) * loss_ce + alpha * loss_kd
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        train_loss.append(epoch_loss / len(train_loader))
        valid_loss = validate(smodel, valid_loader, device)
        
        wandb.log({
                "train_loss": epoch_loss / (step + 1),
                "valid_loss": valid_loss
            })
            
        print(f"[Epoch {epoch + 1}/{n_epochs}] Step {step  + 1}/{len(train_loader)} | loss: {epoch_loss/(step + 1): .3f} | valid loss: {valid_loss: .3f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            save_ckpt(
                    ckpt_path=p.join(ckpt, f"LatentBE_div_model_checkpoint_epoch_{epoch + 1}.pt"), 
                    model=smodel, epoch=epoch + 1, 
                    train_loss=train_loss, best_loss=valid_loss
                    )
            print(f"Success to save checkpoint.")
        else:
            print("No improvement detected. Skipping save")