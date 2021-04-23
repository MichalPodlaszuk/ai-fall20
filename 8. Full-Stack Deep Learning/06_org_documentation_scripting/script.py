import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time

import os
import time

import copy

import requests
import zipfile

from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv(".env")

    if "data.zip" not in os.listdir():
        r = requests.get(
            "https://github.com/polyrand/strive-ml-fullstack-public/blob/main/06_org_documentation_scripting/data.zip?raw=true"
        )

        with open("data.zip", "wb") as f:
            f.write(r.content)

    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall("data")

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    data_dir = "data/hymenoptera_data"
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=4, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(model, criterion, optimizer, scheduler, num_epochs=10):

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), "correct_incorrect.pth")
        return model

    model_conv = torchvision.models.resnet18(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.

    optimizer_sgd = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    optimizer_adam = optim.Adam(model_conv.fc.parameters(), lr=0.001, weight_decay=0.01, eps=1e-5)

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=3, gamma=0.1)

    import argparse
    text = 'This is a really awesome script. It lets you train a model from cli'

    parser = argparse.ArgumentParser(description=text)
    #parser.add_argument('-d', '--download_data', help='downloads the data from specified url', action='store_true')
    parser.add_argument('-t', '--train', help='trains the model', action='store_true')
    parser.add_argument('-e', '--epochs', help='number of epochs', type=int)
    parser.add_argument('-o', '--optimizer', help='optimizer', type=str, choices=['sgd', 'adam'])
    parser.add_argument('-l', '--load_weights', help='load your weights from specified file', action='store_true')
    args = parser.parse_args()

    if args.train:
        start = time.perf_counter()
        if args.optimizer=='sgd':
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_sgd, step_size=3, gamma=0.1)
            model_conv = train_model(
                model_conv, criterion, optimizer_sgd, exp_lr_scheduler, num_epochs=args.epochs
            )
        elif args.optimizer=='adam':
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_adam, step_size=3, gamma=0.1)
            model_conv = train_model(
                model_conv, criterion, optimizer_adam, exp_lr_scheduler, num_epochs=args.epochs
            )
        total = time.perf_counter() - start

        print(f"Total time {total}")