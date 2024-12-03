import torch
import torch.optim as optim

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# 모델 초기화
# W = torch.zeros(1)
# learning rate 설정
# lr = 0.1

# 모델 초기화
W = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W], lr=0.15)

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train * W
    
    # cost gradient 계산
    # cost = torch.mean((hypothesis - y_train) ** 2)
    # gradient = torch.sum((W * x_train - y_train) * x_train)

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(
        epoch, nb_epochs, W.item(), cost.item()
    ))
    # Epoch    0/10 W: 0.000, Cost: 4.666667  
    # Epoch    1/10 W: 1.400, Cost: 0.746666
    # Epoch    2/10 W: 0.840, Cost: 0.119467
    # Epoch    3/10 W: 1.064, Cost: 0.019115
    # Epoch    4/10 W: 0.974, Cost: 0.003058
    # Epoch    5/10 W: 1.010, Cost: 0.000489
    # Epoch    6/10 W: 0.996, Cost: 0.000078
    # Epoch    7/10 W: 1.002, Cost: 0.000013
    # Epoch    8/10 W: 0.999, Cost: 0.000002
    # Epoch    9/10 W: 1.000, Cost: 0.000000
    # Epoch   10/10 W: 1.000, Cost: 0.000000

    # cost gradient로 H(x) 개선
    # W -= lr * gradient

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    