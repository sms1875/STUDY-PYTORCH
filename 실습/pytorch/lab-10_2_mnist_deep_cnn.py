# Lab 11 MNIST and Deep learning CNN
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

# GPU 사용 가능 여부 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 재현성을 위한 랜덤 시드 설정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 하이퍼파라미터 설정
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# MNIST 데이터셋 다운로드 및 로드 (훈련 데이터)
mnist_train = dsets.MNIST(
    root='MNIST_data/',  # 데이터를 저장할 경로
    train=True,  # 훈련 데이터 여부
    transform=transforms.ToTensor(),  # 데이터를 텐서 형태로 변환
    download=True  # 다운로드 여부
)

# MNIST 데이터셋 다운로드 및 로드 (테스트 데이터)
mnist_test = dsets.MNIST(
    root='MNIST_data/',
    train=False,  # 테스트 데이터 여부
    transform=transforms.ToTensor(),
    download=True
)

# 데이터 로더 생성 (훈련 데이터를 배치 단위로 로드)
data_loader = torch.utils.data.DataLoader(
    dataset=mnist_train,
    batch_size=batch_size,
    shuffle=True,  # 데이터를 섞음
    drop_last=True  # 마지막 배치를 버림 (배치 크기가 부족할 때)
)

# CNN 모델 정의
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5  # 드롭아웃 확률

        # L1: Conv -> ReLU -> MaxPool (입력: 28x28x1, 출력: 14x14x32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # L2: Conv -> ReLU -> MaxPool (입력: 14x14x32, 출력: 7x7x64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # L3: Conv -> ReLU -> MaxPool (입력: 7x7x64, 출력: 4x4x128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        # L4: Fully Connected -> ReLU -> Dropout (입력: 4x4x128, 출력: 625)
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)  # Xavier 초기화
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob)  # 드롭아웃 적용
        )

        # L5: Fully Connected (입력: 625, 출력: 10)
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # FC 레이어를 위한 Flatten
        out = self.layer4(out)
        out = self.fc2(out)
        return out

# CNN 모델 생성 및 장치로 이동
model = CNN().to(device)

# 손실 함수 및 옵티마이저 설정
criterion = torch.nn.CrossEntropyLoss().to(device)  # 크로스 엔트로피 손실 함수 (Softmax 포함)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam 옵티마이저

# 모델 학습
total_batch = len(data_loader)  # 총 배치 수
model.train()  # Dropout 활성화
print('Learning started. It takes some time.')

for epoch in range(training_epochs):
    avg_cost = 0  # 에포크당 평균 손실

    for X, Y in data_loader:
        # 입력 데이터를 GPU/CPU 장치로 이동
        X = X.to(device)
        Y = Y.to(device)

        # 옵티마이저 초기화
        optimizer.zero_grad()
        # 모델 예측
        hypothesis = model(X)
        # 손실 계산
        cost = criterion(hypothesis, Y)
        # 손실 역전파
        cost.backward()
        # 가중치 업데이트
        optimizer.step()

        # 배치 손실 누적
        avg_cost += cost / total_batch

    # 에포크 결과 출력
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

print('Learning Finished!')
# [Epoch:    1] cost = 0.190638304
# [Epoch:    2] cost = 0.0532845221
# [Epoch:    3] cost = 0.037660379
# [Epoch:    4] cost = 0.0296484027
# [Epoch:    5] cost = 0.0239125956
# [Epoch:    6] cost = 0.0203637555
# [Epoch:    7] cost = 0.0181394033
# [Epoch:    8] cost = 0.0136777712
# [Epoch:    9] cost = 0.0135163562
# [Epoch:   10] cost = 0.0108031444
# [Epoch:   11] cost = 0.00989781693
# [Epoch:   12] cost = 0.0114726787
# [Epoch:   13] cost = 0.00667280518
# [Epoch:   14] cost = 0.00744756451
# [Epoch:   15] cost = 0.00912202988

# 모델 테스트
with torch.no_grad():  # 기울기 계산 비활성화
    model.eval()  # Dropout 비활성화

    # 테스트 데이터를 모델에 입력
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    # 테스트 데이터 예측
    prediction = model(X_test)
    # 예측값과 실제 라벨 비교
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    # 정확도 계산
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
# Accuracy: 0.9837999939918518