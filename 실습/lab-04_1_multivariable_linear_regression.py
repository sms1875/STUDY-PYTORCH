import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


# 모델 초기화
# W = torch.zeros((3, 1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
model = MultivariateLinearRegressionModel()

hypothesis = model(x_train)

# optimizer 설정
# optimizer = optim.SGD([W, b], lr=1e-5)
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산 1
    # hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # 하지만 x 길이가 1000이라면? 
    # matmul() 을 이용하여 계산

    # H(x) 계산 2
    # hypothesis = x_train.matmul(W) + b # or .mm or @
    prediction = model(x_train)

    # cost 계산
    # cost = torch.mean((hypothesis - y_train) ** 2)
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))
    # Epoch    0/20 Cost: 31667.597656
    # Epoch    1/20 Cost: 9926.266602
    # Epoch    2/20 Cost: 3111.513916
    # Epoch    3/20 Cost: 975.451599
    # Epoch    4/20 Cost: 305.908691
    # Epoch    5/20 Cost: 96.042679
    # Epoch    6/20 Cost: 30.260746
    # Epoch    7/20 Cost: 9.641718
    # Epoch    8/20 Cost: 3.178694
    # Epoch    9/20 Cost: 1.152871
    # Epoch   10/20 Cost: 0.517863
    # Epoch   11/20 Cost: 0.318801
    # Epoch   12/20 Cost: 0.256388
    # Epoch   13/20 Cost: 0.236816
    # Epoch   14/20 Cost: 0.230660
    # Epoch   15/20 Cost: 0.228719
    # Epoch   16/20 Cost: 0.228095
    # Epoch   17/20 Cost: 0.227881
    # Epoch   18/20 Cost: 0.227802
    # Epoch   19/20 Cost: 0.227760
    # Epoch   20/20 Cost: 0.227729