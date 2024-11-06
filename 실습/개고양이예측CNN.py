import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import visdom
import random

print(torch.cuda.is_available())

# === 하이퍼파라미터 및 설정 변수 ===
EPOCHS = 10                 # 에포크 수
IMAGE_SIZE = 128            # 이미지 크기 (가로와 세로)
BATCH_SIZE = 32             # 배치 크기 (None이면 전체 데이터를 한 번에 사용)
LEARNING_RATE = 0.001       # 학습률
NUM_WORKERS = 4             # DataLoader의 num_workers 설정

# Visdom 설정
vis = visdom.Visdom()
vis.close(env="main")
loss_plt = vis.line(Y=torch.tensor([0]), opts=dict(title="Training Loss", xlabel="Iteration", ylabel="Loss"))

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 256)  # 이미지 크기에 맞게 조정
        self.fc2 = nn.Linear(256, 2)  # 고양이와 강아지 이진 분류
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8))  # 이미지 크기에 맞게 조정
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 데이터 경로 설정
train_dir = "/workspaces/STUDY-PYTORCH/kaggle/Cat and Dog/training_set/training_set"
test_dir = "/workspaces/STUDY-PYTORCH/kaggle/Cat and Dog/test_set/test_set"

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 이미지 크기 설정
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화
])

# ImageFolder로 데이터 로드
train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

# DataLoader 설정 (배치 크기를 반영)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE if BATCH_SIZE else len(train_dataset), shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE if BATCH_SIZE else len(test_dataset), shuffle=False, num_workers=NUM_WORKERS)

# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device is {device}')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 클래스 이름 설정
classes = train_dataset.classes  # ["cats", "dogs"]

# 모델 학습
iteration = 0  # 그래프에 표시할 iteration 값
for epoch in range(EPOCHS):
    model.train()
    
    for data in train_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        # 경사 초기화
        optimizer.zero_grad()
        
        # 순전파, 손실 계산, 역전파, 최적화
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Visdom에 손실 값 출력
        iteration += 1
        if iteration % 10 == 0:  # 매 10개 배치마다 업데이트
            vis.line(X=torch.tensor([iteration]), Y=torch.tensor([loss.item()]), win=loss_plt, update="append")

    
    print(f'Epoch {epoch+1}, Loss: {loss.item():.3f}')

print("Finished Training")
# Epoch 1, Loss: 0.755
# Epoch 2, Loss: 0.407
# Epoch 3, Loss: 0.399
# Epoch 4, Loss: 0.275
# Epoch 5, Loss: 0.202
# Epoch 6, Loss: 0.231
# Epoch 7, Loss: 0.117
# Epoch 8, Loss: 0.400
# Epoch 9, Loss: 0.208
# Epoch 10, Loss: 0.009
# Finished Training

# 학습 종료 후 테스트 단계 추가
print("\nEvaluating on test set...")

model.eval()
correct = 0
total = 0

# 테스트 데이터에서 무작위 샘플 이미지 선택
sample_images = None
sample_label = None
sample_pred = None

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 예측 수행
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # 총 개수와 맞춘 개수 계산
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 무작위 이미지 선택 (한 번만 저장)
        if sample_images is None:
            idx = random.randint(0, images.size(0) - 1)
            sample_images = images[idx].cpu()
            sample_label = labels[idx].cpu()
            sample_pred = predicted[idx].cpu()

# 테스트 셋에서의 최종 정확도 계산
accuracy = 100 * correct / total
print(f"Final accuracy of the model on the test images: {accuracy:.2f}%")

# Visdom에 최종 평가 결과 이미지와 정확도 시각화
if sample_images is not None:
    sample_images = (sample_images * 0.5 + 0.5)  # 정규화 해제 (0~1로 변환)
    vis.image(sample_images, opts=dict(title=f"Final Accuracy: {accuracy:.2f}% | True: {classes[sample_label]} | Pred: {classes[sample_pred]}"))

print("Test Evaluation Completed.")
