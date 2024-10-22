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
print("Data head:\n", data.head())
print("\nData shape:", data.shape)
# Data head:
#     User ID    Device Model Operating System  App Usage Time (min/day)  Screen On Time (hours/day)  Battery Drain (mAh/day)  Number of Apps Installed  Data Usage (MB/day)  Age  Gender  User Behavior Class
# 0        1  Google Pixel 5          Android                       393                         6.4                     1872                        67                 1122   40    Male                    4
# 1        2       OnePlus 9          Android                       268                         4.7                     1331                        42                  944   47  Female                    3
# 2        3    Xiaomi Mi 11          Android                       154                         4.0                      761                        32                  322   42    Male                    2
# 3        4  Google Pixel 5          Android                       239                         4.8                     1676                        56                  871   20    Male                    3
# 4        5       iPhone 12              iOS                       187                         4.3                     1367                        58                  988   31  Female                    3

# Data shape: (700, 11)

# 범주형 데이터 인코딩
data['Device Model'] = data['Device Model'].astype('category').cat.codes
data['Operating System'] = data['Operating System'].astype('category').cat.codes
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

print("\nEncoded data head:\n", data.head())
# Encoded data head:
#     User ID  Device Model  Operating System  App Usage Time (min/day)  Screen On Time (hours/day)  Battery Drain (mAh/day)  Number of Apps Installed  Data Usage (MB/day)  Age  Gender  User Behavior Class
# 0        1             0                 0                       393                         6.4                     1872                        67                 1122   40       0                    4
# 1        2             1                 0                       268                         4.7                     1331                        42                  944   47       1                    3
# 2        3             3                 0                       154                         4.0                      761                        32                  322   42       0                    2
# 3        4             0                 0                       239                         4.8                     1676                        56                  871   20       0                    3
# 4        5             4                 1                       187                         4.3                     1367                        58                  988   31       1                    3

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

# 정규화된 데이터 출력
print("\nMean of training data:", mu)
print("Standard deviation of training data:", sigma)
# Mean of training data: tensor([2.0900e+00, 2.2800e-01, 2.7168e+02, 5.2642e+00, 1.5268e+03, 5.0752e+01,
#         9.2623e+02, 3.8560e+01, 4.8400e-01])
# Standard deviation of training data: tensor([1.4415e+00, 4.1996e-01, 1.7850e+02, 3.1153e+00, 8.2903e+02, 2.7258e+01,
#         6.4256e+02, 1.2082e+01, 5.0024e-01])
print("\nNormalized x_train sample:\n", norm_x_train[:5])
# Normalized x_train sample:
#  tensor([[-1.4498, -0.5429,  0.6797,  0.3646,  0.4164,  0.5961,  0.3047,  0.1192,
#          -0.9675],
#         [-0.7561, -0.5429, -0.0206, -0.1811, -0.2361, -0.3211,  0.0276,  0.6986,
#           1.0315],
#         [ 0.6313, -0.5429, -0.6593, -0.4058, -0.9237, -0.6880, -0.9404,  0.2847,
#          -0.9675],
#         [-1.4498, -0.5429, -0.1831, -0.1490,  0.1800,  0.1925, -0.0860, -1.5362,
#          -0.9675],
#         [ 1.3250,  1.8383, -0.4744, -0.3095, -0.1927,  0.2659,  0.0961, -0.6257,
#           1.0315]])

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

print("\nSample predictions (first 10):", predicted_classes[:10].numpy())
print("\nTrue classes (first 10):", y_test[:10].numpy())
# Sample predictions (first 10): [0 3 4 2 4 3 2 0 2 4]
# True classes (first 10): [0 3 4 2 4 3 2 0 2 4]
