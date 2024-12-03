import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

# GPU 사용 가능 여부 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 재현성을 위해 랜덤 시드 설정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 하이퍼파라미터 설정
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# MNIST 데이터셋 다운로드 및 불러오기 (훈련 데이터)
mnist_train = dsets.MNIST(
    root='MNIST_data/',  # 데이터를 저장할 경로
    train=True,  # 훈련 데이터 여부
    transform=transforms.ToTensor(),  # 데이터를 텐서 형태로 변환
    download=True  # 다운로드 여부
)

# MNIST 데이터셋 다운로드 및 불러오기 (테스트 데이터)
mnist_test = dsets.MNIST(
    root='MNIST_data/',
    train=False,  # 테스트 데이터 여부
    transform=transforms.ToTensor(),
    download=True
)

# 데이터 로더 생성 (훈련 데이터셋을 배치 단위로 로드)
data_loader = torch.utils.data.DataLoader(
    dataset=mnist_train,  # 데이터셋
    batch_size=batch_size,  # 배치 크기
    shuffle=True,  # 데이터를 섞을지 여부
    drop_last=True  # 마지막 배치를 버릴지 여부 (배치 크기가 부족할 때)
)

# CNN 모델 정의
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫 번째 컨볼루션 레이어 (입력: 1채널, 출력: 32채널, 커널 크기: 3x3)
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 출력 크기: 28x28x32
            torch.nn.ReLU(),  # 활성화 함수
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 풀링 후 크기: 14x14x32
        )
        # 두 번째 컨볼루션 레이어 (입력: 32채널, 출력: 64채널, 커널 크기: 3x3)
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 출력 크기: 14x14x64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 풀링 후 크기: 7x7x64
        )
        # 완전 연결 레이어 (입력: 7x7x64, 출력: 10)
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)
        # Xavier 초기화로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # 첫 번째 레이어 통과
        out = self.layer1(x)
        # 두 번째 레이어 통과
        out = self.layer2(out)
        # 완전 연결 레이어를 위해 텐서를 평탄화
        out = out.view(out.size(0), -1)
        # 완전 연결 레이어 통과
        out = self.fc(out)
        return out

# CNN 모델 인스턴스화 및 장치로 이동
model = CNN().to(device)

# 손실 함수 및 최적화 알고리즘 정의
criterion = torch.nn.CrossEntropyLoss().to(device)  # 크로스 엔트로피 손실 함수 (내부적으로 Softmax 포함)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam 옵티마이저

# 모델 학습
total_batch = len(data_loader)  # 총 배치 수
print('Learning started. It takes sometime.')

for epoch in range(training_epochs):
    avg_cost = 0  # 에포크당 평균 손실

    for X, Y in data_loader:
        # 입력 데이터와 라벨 데이터를 장치로 이동
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
        # 모델 매개변수 업데이트
        optimizer.step()

        # 배치 손실을 평균 손실에 더함
        avg_cost += cost / total_batch

    # 에포크마다 손실 출력
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

print('Learning Finished!')
# [Epoch:    1] cost = 0.225939184
# [Epoch:    2] cost = 0.0630982518
# [Epoch:    3] cost = 0.0463336967
# [Epoch:    4] cost = 0.0374494195
# [Epoch:    5] cost = 0.0313653052
# [Epoch:    6] cost = 0.0261803828
# [Epoch:    7] cost = 0.0218190774
# [Epoch:    8] cost = 0.0183058828
# [Epoch:    9] cost = 0.0164249353
# [Epoch:   10] cost = 0.0131303286
# [Epoch:   11] cost = 0.00988870859
# [Epoch:   12] cost = 0.00991326477
# [Epoch:   13] cost = 0.00879074447
# [Epoch:   14] cost = 0.0067628026
# [Epoch:   15] cost = 0.00727953995

# 모델 테스트 및 정확도 계산
with torch.no_grad():  # 테스트 단계에서는 기울기 계산 비활성화
    # 테스트 데이터를 모델에 입력하기 위한 전처리
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    # 테스트 데이터에 대한 예측
    prediction = model(X_test)
    # 예측 값과 실제 라벨 비교
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    # 정확도 계산
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
# Accuracy: 0.9873999953269958