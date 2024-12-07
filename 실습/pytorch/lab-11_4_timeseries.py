import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # GUI 백엔드 설정
import matplotlib.pyplot as plt
# 랜덤 시드 설정
torch.manual_seed(0)

# 데이터 스케일링 함수 (0과 1 사이로 정규화)
def minmax_scaler(data):
    numerator = data - np.min(data, 0)  # 각 열에서 최솟값을 뺌
    denominator = np.max(data, 0) - np.min(data, 0)  # 각 열의 범위를 계산
    return numerator / (denominator + 1e-7)  # 1e-7을 더해 나눗셈 에러 방지

# 데이터셋 생성 함수
def build_dataset(time_series, seq_length):
    dataX = []  # 입력 데이터 (X)
    dataY = []  # 출력 데이터 (Y)
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]  # 시퀀스 길이만큼 데이터를 자름
        _y = time_series[i + seq_length, [-1]]  # 다음 날의 종가를 예측
        print(_x, "->", _y)
        dataX.append(_x)  # 입력 데이터 추가
        dataY.append(_y)  # 출력 데이터 추가
    return np.array(dataX), np.array(dataY)

# 하이퍼파라미터 설정
seq_length = 7  # 시퀀스 길이 (7일)
data_dim = 5  # 입력 데이터 차원 (시가, 고가, 저가, 종가, 거래량)
hidden_dim = 10  # LSTM 히든 노드 개수
output_dim = 1  # 출력 차원 (종가)
learning_rate = 0.01  # 학습률
iterations = 500  # 학습 반복 횟수

# 데이터 로드
xy = np.loadtxt("data-02-stock_daily.csv", delimiter=",")  # CSV 파일에서 데이터 로드
xy = xy[::-1]  # 데이터를 시간 순서대로 정렬 (역순)

# 학습 데이터와 테스트 데이터로 분리
train_size = int(len(xy) * 0.7)  # 학습 데이터는 전체의 70%
train_set = xy[0:train_size]  # 학습 데이터
test_set = xy[train_size - seq_length:]  # 테스트 데이터 (시퀀스 길이를 고려)

# 데이터 스케일링 (0과 1 사이로 정규화)
train_set = minmax_scaler(train_set)
test_set = minmax_scaler(test_set)

# 학습 데이터와 테스트 데이터셋 생성
trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

# 텐서로 변환 (PyTorch가 처리할 수 있는 형식)
trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)
testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)

# LSTM 모델 정의
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)  # LSTM 레이어
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)  # Fully Connected Layer

    def forward(self, x):
        x, _status = self.rnn(x)  # LSTM 출력
        x = self.fc(x[:, -1])  # 마지막 시점의 출력만 사용
        return x

# 모델 초기화
net = Net(data_dim, hidden_dim, output_dim, 1)

# 손실 함수와 옵티마이저 설정
criterion = torch.nn.MSELoss()  # 손실 함수: 평균 제곱 오차(MSE)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 옵티마이저: Adam

# 모델 학습
for i in range(iterations):
    optimizer.zero_grad()  # 기울기 초기화
    outputs = net(trainX_tensor)  # 모델 예측
    loss = criterion(outputs, trainY_tensor)  # 손실 계산
    loss.backward()  # 역전파
    optimizer.step()  # 가중치 업데이트
    print(i, loss.item())  # 학습 진행 상황 출력
# 0 0.2271038144826889
# 1 0.18340934813022614
# 2 0.15106436610221863
# 3 0.1261231154203415
# ...
# 496 0.001276303199119866
# 497 0.001275834976695478
# 498 0.0012753689661622047
# 499 0.0012749056331813335

# 테스트 결과 시각화
plt.plot(testY)  # 실제 종가
plt.plot(net(testX_tensor).data.numpy())  # 예측된 종가
plt.legend(['original', 'prediction'])  # 범례 추가
plt.show()
