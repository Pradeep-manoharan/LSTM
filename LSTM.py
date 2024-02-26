# Setup & library

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

# Hyper-Parameters

input_size = 28
hidden_size = 128  # Number of hidden size in the hidden layer
num_layer = 2  # Number of LSTM
num_classes = 10
sequence_length = 28
Learning_rate = 0.01
batch_size = 64
num_epochs = 1

# Dataset Preparation

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# Define the model

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, number_class):
        super(LSTMModel,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # set Initial hidden  and cell states
        ho = torch.zeros(num_layer, x.size(0), hidden_size)
        co = torch.zeros(num_layer, x.size(0), hidden_size)

        # Forward pass in LSTM

        out,_ = self.lstm(x, (ho, co))

        out = self.fc1(out[:, -1, :])

        return out


model = LSTMModel(input_size, hidden_size, num_layer, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)

# Training Data

for epoch in range(num_epochs):
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data[:, 0, :, :]
        data = data.reshape(-1, sequence_length, input_size)

        # Forward pass

        output = model(data)
        loss = criterion(output, label)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(batch_idx+1) % 100 == 0:
            print(f'Epochs [{epoch+1}/{num_epochs}],steps[{batch_idx+1}/{len(train_loader)}],loss ={loss.item():.4f}')


# Test Model

model.eval()

with torch.no_grad():
    correct = 0
    total = 0

    for data, target in test_loader:
        data = data[:,0:,:]
        data = data.reshape(-1,sequence_length,input_size)


        output= model(data)

        _,predicted = torch.max(output,-1)
        total += target.size(0)
        correct+= (predicted==target).sum().item()

accuracy = 100 * correct / total

print(f'Accuracy = {accuracy}')