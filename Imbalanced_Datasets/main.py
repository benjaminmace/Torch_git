import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomPerspective(0.25),
            transforms.RandomRotation(20, resample=Image.BILINEAR),
            transforms.ToTensor(),
            #transforms.Normalize([0.5459, 0.4944, 0.4324], [0.2299, 0.2196, 0.2226])
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(4 * (1/len(files)))

    sample_weights = [0] * len(dataset)

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler,)

    return loader

mean_dataset = datasets.ImageFolder(root='dataset', transform=transforms.ToTensor())
mean_loader = DataLoader(mean_dataset)

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=2, stride=1, padding=0, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
    nn.Flatten(),
    nn.Linear(4096, 8192),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(8192, 1024),
    nn.ReLU(),
    nn.Linear(1024, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
).to(device=device)

from torchsummary import summary
#summary(model, (3, 224, 224))

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

loader = get_loader(root_dir='dataset', batch_size=8)

def fit(epochs, model, criterion, optimizer, loader):
    for epoch in range(epochs):
        for data, targets in loader:
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data).to(device)
            loss = criterion(scores, targets).to(device)
            print(f'Epoch {epoch}: Loss: {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(loss)
    torch.save(model, 'unbalanced_2.pth')


fit(25, model, criterion, optimizer, loader)


