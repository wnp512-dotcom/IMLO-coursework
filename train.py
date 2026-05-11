import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize
import matplotlib.pyplot as plt

#Load the training dataser
training_data = datasets.OxfordIIITPet(
    root="data",
    split="trainval",
    download=True,
    transform= transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
    ])
)

#Load the test dataset
test_data = datasets.OxfordIIITPet(
    root="data",
    split="test",
    download=True,
    transform= transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
    ])
)

#Map each category to a name
labels_map = {
    0: "Abyssinian",
    1: "American Bulldog",
    2: "American Pit Bull Terrier",
    3: "Basset Hound",
    4: "Beagle",
    5: "Bengal",
    6: "Birman",
    7: "Bombay",
    8: "Boxer",
    9: "British Shorthair",
    10: "Chihuahua",
    11: "Egyptian Mau",
    12: "English Cocker Spaniel",
    13: "English Setter",
    14: "German Shorthaired",
    15: "Great Pyrenees",
    16: "Havanese",
    17: "Japanese Chin",
    18: "Keeshond",
    19: "Leonberger",
    20: "Maine Coon",
    21: "Miniature Pinscher",
    22: "Newfoundland",
    23: "Persian",
    24: "Pomeranian",
    25: "Pug",
    26: "Ragdoll",
    27: "Russian Blue",
    28: "Saint Bernard",
    29: "Samyoed",
    30: "Scottish Terrier",
    31: "Shiba Inu",
    32: "Siamese",
    33: "Sphynx",
    34: "Staffordshire Bull Terrier",
    35: "Wheaten Terrier",
    36: "Yorkshire Terrier",
}

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 8, 17)
    self.conv2 = nn.Conv2d(8, 18, 5)
    self.conv3 = nn.Conv2d(18, 32, 3)
    self.fc1 = nn.Linear(32 * 6 * 6, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 37)

  def forward(self, x):
      x = F.max_pool2d(F.relu(self.conv1(x)), 4)
      x = F.max_pool2d(F.relu(self.conv2(x)), 4)
      x = F.max_pool2d(F.relu(self.conv3(x)), 2)
      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x
  
net = NeuralNetwork()

learning_rate = 1e-3
batch_size = 64
epochs = 5
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def train_loop(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, net, criterion, optimizer)
print("Done!")