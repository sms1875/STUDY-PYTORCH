import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 재현성을 위해 시드 고정
torch.manual_seed(1)

# Softmax 예제
# 숫자를 확률로 변환하기 위해 소프트맥스를 사용
z = torch.FloatTensor([1, 2, 3])
# 소프트맥스를 사용하여 각 요소를 확률로 변환
hypothesis = F.softmax(z, dim=0)
print(hypothesis)

# 확률의 합이 1인지 확인
print(hypothesis.sum())

# Cross Entropy Loss (Low-level)
# 다중 클래스 분류를 위한 손실 함수로 교차 엔트로피 손실을 사용

# 임의의 값으로 텐서 생성
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)

# 정답(y) 생성
y = torch.randint(5, (3,)).long()
print(y)

# y를 원-핫 인코딩 형태로 변환
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y_one_hot)

# 교차 엔트로피 손실을 계산
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

# PyTorch에서 제공하는 고수준의 Cross Entropy Loss 함수들

# log_softmax를 사용한 저수준 계산
print(torch.log(F.softmax(z, dim=1)))

# log_softmax를 사용한 고수준 계산
print(F.log_softmax(z, dim=1))

# nll_loss 사용
print((y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean())
print(F.nll_loss(F.log_softmax(z, dim=1), y))

# cross_entropy를 사용하여 log_softmax와 nll_loss 결합
print(F.cross_entropy(z, y))

# Low-level Cross Entropy Loss로 훈련
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # 예측값 계산 (소프트맥스 적용)
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
    
    # 원-핫 인코딩 변환
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    
    # 교차 엔트로피 손실 계산
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    # 비용으로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

# F.cross_entropy를 사용하여 훈련
# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # 예측값 계산
    z = x_train.matmul(W) + b
    
    # cross_entropy를 사용한 비용 계산
    cost = F.cross_entropy(z, y_train)

    # 비용으로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

# nn.Module을 사용하여 고수준 구현
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # 입력 4, 출력 클래스 3

    def forward(self, x):
        return self.linear(x)

# 모델 인스턴스 생성
model = SoftmaxClassifierModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # 예측값 계산
    prediction = model(x_train)

    # 비용 계산
    cost = F.cross_entropy(prediction, y_train)

    # 비용으로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
