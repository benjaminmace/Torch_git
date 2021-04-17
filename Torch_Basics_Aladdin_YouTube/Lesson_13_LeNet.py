import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformers = transforms.Compose(
    [
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.100], std=[0.2752])
    ]
)

in_channels = 1
num_classes = 10
learning_rate = 0.0001
batch_size = 64
num_epochs = 8

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=120,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(0, 0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)

        return x

train_dataset = datasets.MNIST(root='Dataset/', train=True, transform=transformers, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='Dataset/', train=False, transform=transformers, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

def get_mean_std(loader):
    channels_sum, channels_sqr_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqr_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum/num_batches
    std = (channels_sqr_sum/num_batches - mean**2)**0.5

    return mean, std


mean, std = get_mean_std(train_loader)

model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):

        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)

        loss = criterion(scores, targets)
        print(epoch, loss)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)





