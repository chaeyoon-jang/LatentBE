import torch
import wandb
from datetime import datetime
import torchvision
from torchvision import datasets, transforms
from .utils import set_seed
from .cnn_be import CNN_be
from .cnn import CNN 
from .train import train
from .train_KD import train_kd
from .train_latentBE_div import train_latentbe_div
from .train_latentBE import train_latentbe
from .option import get_arg_parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"-- Running on {device}. -- ")
    
    # Set seed
    set_seed(args.seed)
    
    # Loading data    
    print("Loading dataloaders...")
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    train_dataset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    test_dataset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    
    train_size = int(0.8*len(train_dataset))
    valid_size = len(train_dataset) - train_size
    
    generator = torch.Generator()
    generator.manual_seed(0)
    
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size], generator=generator)
    
    print("Train dataset : "+str(len(train_dataset)))
    print("Valid dataset : "+str(len(valid_dataset)))
    print("Test dataset : "+str(len(test_dataset)))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("Finish. \n")
    
    # Loading model
    print("Get model...")
    if args.mode == "KD":
        tmodel = []
        for _ in range(args.ensemble_size):
            model = CNN()
            model = model.to(device)
            tmodel.append(model)
        smodel = CNN()
        smodel.to(device)
        
        print("Start student model training...")
        wandb.init(
            project = "fashion_mnist",
            entity = "chaeyoon-jang"
        )
        wandb.config.update(args)
        wandb.run.name = datetime.now().strftime('KD %Y-%m-%d %H:%M')
        
        print("Start train")
        
        train_kd(
            tmodel=tmodel,
            smodel=smodel,
            train_loader=train_loader,
            valid_loader=valid_loader,
            n_epochs = args.n_epochs,
            device = device,
            learning_rate = args.learning_rate
        )
        
        print("Finish.\n")
        
    elif args.mode == "latentBE":
        tmodel = []
        for _ in range(args.ensemble_size):
            model = CNN()
            model = model.to(device)
            tmodel.append(model)
        smodel = CNN_be(bias_is=True)
        smodel.to(device)
        
        print("Start student model trainig...")
        wandb.init(
            project = "fashion_mnist",
            entity = "chaeyoon-jang"
        )
        wandb.config.update(args)
        wandb.run.name = datetime.now().strftime('LatentBE %Y-%m-%d %H:%M')
        
        print("Start train")
        
        train_latentbe(
            tmodel=tmodel,
            smodel=smodel,
            train_loader=train_loader,
            valid_loader=  valid_loader,
            n_epochs = args.n_epochs,
            device = device,
            learning_rate = args.learning_rate
        )
        
        print("Finish.\n")
        
    elif args.mode == "latentBE_div":
        tmodel = []
        for _ in range(args.ensemble_size):
            model = CNN()
            model = model.to(device)
            tmodel.append(model)
        smodel = CNN_be(bias_is=True)
        smodel.to(device)
        
        print("Start student model trainig...")
        wandb.init(
            project = "fashion_mnist",
            entity = "chaeyoon-jang"
        )
        wandb.config.update(args)
        wandb.run.name = datetime.now().strftime('LatentBE+div %Y-%m-%d %H:%M')
        
        print("Start train")
        
        train_latentbe_div(
            tmodel=tmodel,
            smodel=smodel,
            train_loader=train_loader,
            valid_loader=  valid_loader,
            n_epochs = args.n_epochs,
            device = device,
            learning_rate = args.learning_rate
        )
        
        print("Finish.\n")
        
    else:
        model = CNN()
        model.to(device)
        
        print("Start teacher model trainig...")
        wandb.init(
            project = "fashion_mnist",
            entity = "chaeyoon-jang"
        )
        wandb.config.update(args)
        wandb.run.name = datetime.now().strftime('Teacher model %Y-%m-%d %H:%M')
        
        print("Start train")
        
        train(
            model=model,
            train_loader=train_loader,
            valid_loader=  valid_loader,
            n_epochs = args.n_epochs,
            device = device,
            learning_rate = args.learning_rate,
            seed = args.seed
        )
        
        print("Finish.\n")
        
if __name__ == '__main__':
    main()