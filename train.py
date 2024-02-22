import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
# from collections import OrderedDict
# from PIL import Image
import datetime

def args_input():
    parser = argparse.ArgumentParser(description='Defining Training Parameters')

    parser.add_argument('-d', '--dir', default='flowers', type=str, 
                        help='Target path of data directory')
    parser.add_argument('-a', '--arch', type=str, default='vgg16', 
                        help='selecting network architecture available on pyTorch')
    parser.add_argument('-g', '--gpu', action='store_true', default=False, 
                        help='if gpu flag is not provided, the value will be False which means using cpu')
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-hu', '--hidden-units', type=int, 
                        help='the number of hidden units in classifier')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-sp', '--savepoint', type=str, default='train_checkpoint.pth', 
                        help='filename for saving the training checkpoint, so can be loaded afterward')
    args = parser.parse_args()
    return args

def load_data(dir):
    train_dir = dir + '/train'
    valid_dir = dir + '/valid'
    test_dir = dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    val_test_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_datasets = datasets.ImageFolder(valid_dir, transform=val_test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    train_loaders = DataLoader(train_datasets, batch_size=64, shuffle=True)
    val_loaders = DataLoader(val_datasets, batch_size=64)
    test_loader = DataLoader(test_datasets, batch_size=64)

    return train_loaders, val_loaders, train_datasets

def build_model(arch, hidden_units=None):
    if hidden_units is None:
        hidden_units=4096

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(nn.Linear(25088, hidden_units), 
                                   nn.ReLU(), nn.Dropout(p=0.5), 
                                   nn.Linear(hidden_units, 512), 
                                   nn.ReLU(), 
                                   nn.Dropout(p=0.5), 
                                   nn.Linear(512, 102), 
                                   nn.LogSoftmax(dim=1))
    elif arch == "densenet":
        model = models.densenet121(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(nn.Linear(25088, hidden_units), 
                                   nn.ReLU(), nn.Dropout(p=0.5), 
                                   nn.Linear(hidden_units, 512), 
                                   nn.ReLU(), 
                                   nn.Dropout(p=0.5), 
                                   nn.Linear(512, 102), 
                                   nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    return model

def train_model(model, train_loaders, val_loaders, criterion, optimizer, learning_rate, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    running_loss = 0
    running_accuracy = 0
    print_every = 25

    for epoch in range(epochs):
        steps = 0
        
        model.train()
        
        for inputs, labels in train_loaders:
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            train_loss = criterion(logps, labels)
            train_loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            running_loss+=train_loss.item()
            
            if steps % print_every == 0:
                val_loss = 0
                val_accuracy = 0
                
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in val_loaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        val_loss+=batch_loss.item()
                        
                        # Calculate val accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f'Epoch {epoch+1}/{epochs} | Step {steps}')
                print(f'Train Loss: {running_loss/print_every:.3f} | Train Accuracy: {running_accuracy/print_every*100:.2f}%')
                print(f'Valid Loss: {val_loss/len(val_loaders):.3f} | Valid Accuracy: {val_accuracy/len(val_loaders)*100:.2f}%')
                
                # reset the hyperparam to 0
                running_loss = 0
                running_accuracy = 0

    print("Training has done")
    return model, criterion    

def test(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model.eval()

    criterion = nn.NLLLoss()

    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss+=batch_loss.item()
            
            # Calculate test accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
    print(f'Test Loss: {test_loss/len(test_loader):.3f}')
    print(f'Test Accuracy: {test_accuracy/len(test_loader)*100:.3f}%')
        
def save_checkpoint(model, arch, filename_prefix, train_datasets):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.pth"

    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'arch': arch,
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_datasets.class_to_idx}

    torch.save(checkpoint, filename)
    print(f"Checkpoint was created: {filename}")

def main():
    
    args = args_input()
    model = build_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
   
    train_loaders, val_loaders, train_datasets = load_data(args.dir) 
       
    train_model(model, train_loaders, val_loaders, criterion, optimizer, args.learning_rate, args.epochs)
    
    save_checkpoint(model, args.arch, "checkpoint", train_datasets) 
    

if __name__ == "__main__":
    main()
