import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# For reproducibility
torch.manual_seed(1)

# 모델 class
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9, 5)
    def forward(self, x):
        return self.linear(x)
    
# 데이터 로드
data = pd.read_csv('user_behavior_dataset.csv')

# 범주형 데이터 인코딩
data['Device Model'] = data['Device Model'].astype('category').cat.codes
data['Operating System'] = data['Operating System'].astype('category').cat.codes
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# User Behavior Class를 1~5에서 0~4로 변환
data['User Behavior Class'] = data['User Behavior Class'] - 1

# 입력데이터와 정답 분리
x_data = data.drop(['User ID', 'User Behavior Class'], axis=1).values  # User ID, 정답 제외
y_data = data['User Behavior Class'].values

# 학습 데이터 (1~500)
x_train = torch.FloatTensor(x_data[:500, :]) 
y_train = torch.LongTensor(y_data[:500])

# 테스트 데이터 (501~700)
x_test = torch.FloatTensor(x_data[500:700, :]) 
y_test = torch.LongTensor(y_data[500:700])

# 데이터 정규화
mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
norm_x_train = (x_train - mu) / sigma
norm_x_test = (x_test - mu) / sigma  # x_train의 평균과 표준편차로 x_test 정규화

# 모델 초기화
model = SoftmaxClassifierModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-1)

# 학습 과정
nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    prediction = model(norm_x_train)
    
    # cost 계산
    cost = nn.CrossEntropyLoss()(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')

# 모델 평가
with torch.no_grad():
    test_prediction = model(norm_x_test)
    predicted_classes = torch.argmax(test_prediction, dim=1)
    accuracy = (predicted_classes == y_test).float().mean()
    print(f'Accuracy: {accuracy.item() * 100:.2f}%')

# lr 클수록 낮아짐, Epoch 클수록 낮아짐
# Epoch    0/2000 Cost: 1.474506
# Epoch  100/2000 Cost: 0.754614
# Epoch  200/2000 Cost: 0.621768
# Epoch  300/2000 Cost: 0.541334
# Epoch  400/2000 Cost: 0.484654
# Epoch  500/2000 Cost: 0.441403
# Epoch  600/2000 Cost: 0.406742
# Epoch  700/2000 Cost: 0.378032
# Epoch  800/2000 Cost: 0.353684
# Epoch  900/2000 Cost: 0.332663
# Epoch 1000/2000 Cost: 0.314264
# Epoch 1100/2000 Cost: 0.297980
# Epoch 1200/2000 Cost: 0.283436
# Epoch 1300/2000 Cost: 0.270347
# Epoch 1400/2000 Cost: 0.258492
# Epoch 1500/2000 Cost: 0.247694
# Epoch 1600/2000 Cost: 0.237811
# Epoch 1700/2000 Cost: 0.228726
# Epoch 1800/2000 Cost: 0.220342
# Epoch 1900/2000 Cost: 0.212580
# Epoch 2000/2000 Cost: 0.205369
# Accuracy: 99.50%