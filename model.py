import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import os
from PIL import Image
class AttractionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AttractionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def Process():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_data = ImageFolder(root='./data/train', transform=transform)
    test_data = ImageFolder(root='./data/test', transform=transform)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
    model = AttractionClassifier(num_classes=len(train_data.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  
            outputs = model(images) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy on test set: {accuracy:.2%}")
    torch.save(model.state_dict(), 'attraction_classifier.pth')
def classify_new_image(img):
    model_file = 'attraction_classifier.pth'
    if not os.path.exists(model_file):
        Process()
    model = AttractionClassifier(num_classes=4)
    model.load_state_dict(torch.load('attraction_classifier.pth'))
    model.eval()  
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    image = transform(img).unsqueeze(0) 
    classes = ['beach','food','mountain','park']
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = classes[predicted.item()]
    return predicted_label
try:
    print(classify_new_image(Image.open('attraction.jpeg').convert('RGB')))
except Exception as e:
    print(f"Error loading image: {e}")