import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# For reproducibility
torch.manual_seed(1)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1) # self.linear = {W, b} , m = ?, d = 8
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
    
model = BinaryClassifier()

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32) 
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


W = torch.zeros((8, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
# optimizer = optim.SGD([W, b], lr=1)
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 100
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # Cost 계산
    # hypothesis = torch.sigmoid(x_train.matmul(W) + b) # or .mm or @
    # cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))
        # Epoch    0/100 Cost: 0.704829 Accuracy 45.72%
        # Epoch   10/100 Cost: 0.572391 Accuracy 67.59%
        # Epoch   20/100 Cost: 0.539563 Accuracy 73.25%
        # Epoch   30/100 Cost: 0.520042 Accuracy 75.89%
        # Epoch   40/100 Cost: 0.507561 Accuracy 76.15%
        # Epoch   50/100 Cost: 0.499125 Accuracy 76.42%
        # Epoch   60/100 Cost: 0.493177 Accuracy 77.21%
        # Epoch   70/100 Cost: 0.488846 Accuracy 76.81%
        # Epoch   80/100 Cost: 0.485612 Accuracy 76.28%
        # Epoch   90/100 Cost: 0.483146 Accuracy 76.55%
        # Epoch  100/100 Cost: 0.481234 Accuracy 76.81%