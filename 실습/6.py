import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 모델 class
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 7)
    def forward(self, x):
        return self.linear(x)
       
# 데이터
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_train = torch.FloatTensor(xy[:, 0:-1])
y_train = torch.LongTensor(xy[:, [-1]]).squeeze()

# 모델 초기화
model = SoftmaxClassifierModel()

# For reproducibility
torch.manual_seed(1)

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
        # Epoch    0/1000 Cost: 2.167349
        # Epoch  100/1000 Cost: 0.478046
        # Epoch  200/1000 Cost: 0.322867
        # Epoch  300/1000 Cost: 0.249685
        # Epoch  400/1000 Cost: 0.204888
        # Epoch  500/1000 Cost: 0.174191
        # Epoch  600/1000 Cost: 0.151702
        # Epoch  700/1000 Cost: 0.134461
        # Epoch  800/1000 Cost: 0.120799
        # Epoch  900/1000 Cost: 0.109696
        # Epoch 1000/1000 Cost: 0.100488