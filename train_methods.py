import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from collections import OrderedDict
from torchvision import datasets, transforms


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    training_data_transforms_only = transforms.Compose([transforms.RandomRotation(15),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])])

    data_transforms = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets_train = datasets.ImageFolder(train_dir, transform=training_data_transforms_only)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform=data_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    dataloader_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size=32)
    
    return dataloader_train, dataloader_valid, image_datasets_train.class_to_idx


def calculate_accuracy(model, criterion, dataloader, gpu):
    model.eval()
    correct = 0
    loss = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            if gpu == 'gpu' and torch.cuda.is_available():
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if criterion:
                loss += criterion(outputs, labels).item()

    return loss, 100*correct / total


def NeuralNetwork(arch, hidden_units, learning_rate, gpu, input_size=25088, output_size=102, dropout=0.2):
    # Load pre-trained network
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'squeezenet1_0':
        model = models.squeezenet1_0(pretrained=True)
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif arch == 'inception_v3':
        model = models.inception_v3(pretrained=True)
    else:
        print("{} pretrained model does not exist; therefore, vgg16 is used instead.".format(arch))
        model = models.vgg16(pretrained=True)
        
    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(input_size, hidden_units)),
                        ('relu1', nn.ReLU()),
                        ('dropout', nn.Dropout2d(p=dropout)),
                        ('output', nn.Linear(hidden_units, output_size)),
                        ('logsoftmax', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier

    if gpu == 'gpu' and torch.cuda.is_available():
        # change to cuda
        model.to('cuda')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer


def train_classifier(epochs, gpu, model, criterion, optimizer, dataloader_train, dataloader_valid):
    print_every = 30
    steps = 0
    
    model.train()
    for e in range(epochs):
        running_loss = 0
        for _, (inputs, labels) in enumerate(dataloader_train):
            steps += 1
            
            if gpu == 'gpu' and torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, valid_accuracy = calculate_accuracy(model, criterion, dataloader_valid, gpu)
                print("Epoch: {}/{} ".format(e+1, epochs),
                      "Training Loss: {:.3f} ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} ".format(valid_loss/print_every),
                      "Validation Accuracy: {:.2f}%".format(valid_accuracy))
                running_loss = 0
                model.train()

                print("Epoch {} finished!".format(e+1))
    

def train(args):
    input_size = 25088
    output_size = 102
    data_dir = args.data_dir
    arch = args.arch
    hidden_units = int(args.hidden_units)
    learning_rate = float(args.learning_rate)
    gpu = args.gpu
    epochs = int(args.epochs)
    save_dir = args.save_dir

    dataloader_train, dataloader_valid, class_to_idx = load_data(data_dir)
    
    model, criterion, optimizer = NeuralNetwork(arch, hidden_units, learning_rate, gpu, input_size, output_size)
    
    train_classifier(epochs, gpu, model, criterion, optimizer, dataloader_train, dataloader_valid)
    
    # save checkpoint
    checkpoint = {
        'arch': arch,
        'input_size': input_size,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'output_size': output_size,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, save_dir)
    print("{} saved successfully!".format(save_dir))
    
    