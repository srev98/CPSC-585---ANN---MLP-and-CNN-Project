import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

### MLP architecture
class MLP(nn.Module):
    def __init__(self, num_input, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

### CNN architecture
class CNN(nn.Module):
    def __init__(self, dropout_pr, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.dropout = nn.Dropout(dropout_pr)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    return train_loss, train_accuracy

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total

    return test_loss, test_accuracy

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    #Looding data
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    mlp_model = MLP(784, 64, 10).to(device)
    cnn_model = CNN(0.25, 10).to(device)

    criterion = nn.CrossEntropyLoss()
    mlp_optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)
    cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9)

    # MLP training
    import time

    start_time = time.time()
    for epoch in range(5):
        mlp_train_loss, mlp_train_acc = train_model(mlp_model, train_loader, mlp_optimizer, criterion, device)
        print(f'MLP: Epoch [{epoch + 1}/5], Loss: {mlp_train_loss:.4f}, Accuracy: {mlp_train_acc:.2f}%')

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time for MLP model: {training_time:.2f} seconds")

    # MLP testing
    mlp_test_loss, mlp_test_acc = test_model(mlp_model, test_loader, criterion, device)
    print(f'MLP: Test Loss: {mlp_test_loss:.4f}, Test Accuracy: {mlp_test_acc:.2f}%')

    # CNN training
    start_time2 = time.time()
    for epoch in range(5):
        cnn_train_loss, cnn_train_acc = train_model(cnn_model, train_loader, cnn_optimizer, criterion, device)
        print(f'CNN: Epoch [{epoch + 1}/5], Loss: {cnn_train_loss:.4f}, Accuracy: {cnn_train_acc:.2f}%')
    
    end_time2 = time.time()
    training_time2 = end_time2 - start_time2
    print(f"Training time for MLP model: {training_time2:.2f} seconds")

    # CNN testing
    cnn_test_loss, cnn_test_acc = test_model(cnn_model, test_loader, criterion, device)
    print(f'CNN: Test Loss: {cnn_test_loss:.4f}, Test Accuracy: {cnn_test_acc:.2f}%')

if __name__ == "__main__":
    main()
