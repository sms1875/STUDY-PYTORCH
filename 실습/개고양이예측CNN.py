import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # (이미지 크기 128x128 기준)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 데이터 경로 설정 (로컬 경로)
train_dir = "/workspaces/STUDY-PYTORCH/kaggle/Cat and Dog/training_set/training_set"
test_dir = "/workspaces/STUDY-PYTORCH/kaggle/Cat and Dog/test_set/test_set"

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 이미지 크기 통일
    transforms.ToTensor(),          # 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화
])

# ImageFolder로 데이터 로드
train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# 모델 초기화 및 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 경사 초기화
        optimizer.zero_grad()
        
        # 순전파, 손실 계산, 역전파, 최적화
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

print("Finished Training")
# Epoch 1/10, Loss: 0.6654650682494754
# Epoch 2/10, Loss: 0.5780091552980362
# Epoch 3/10, Loss: 0.5247365792119314
# Epoch 4/10, Loss: 0.46941256523132324
# Epoch 5/10, Loss: 0.4167997170062292
# Epoch 6/10, Loss: 0.352424107787628
# Epoch 7/10, Loss: 0.2809536666387603
# Epoch 8/10, Loss: 0.20615432639088896
# Epoch 9/10, Loss: 0.15371812350811467
# Epoch 10/10, Loss: 0.10207413670621694

# 모델 평가
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 예측 수행
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the model on the test images: {accuracy:.2f}%")
# Accuracy of the model on the test images: 78.30%