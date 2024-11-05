import torch
import torch.optim as optim

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]]) # 입력
y_train = torch.FloatTensor([[2], [4], [6]]) # 출력

# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# W,와 b를 학습시킬 것 이라고 pytorch에 알림

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01) # 학습시킬 두 개의 데이터를 리스트로 만들어 넣음, 적당한 running rate 넣음.

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train * W + b
    
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad() # zero_grad()로 gradient 초기화
    cost.backward() # backward()로 gradient 계산
    optimizer.step() # step() 으로 계산된 gradient 방향으로 W와 b 를 개선

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
        # Epoch    0/1000 W: 0.187, b: 0.080 Cost: 18.666666
        # Epoch  100/1000 W: 1.746, b: 0.578 Cost: 0.048171
        # Epoch  200/1000 W: 1.800, b: 0.454 Cost: 0.029767
        # Epoch  300/1000 W: 1.843, b: 0.357 Cost: 0.018394
        # Epoch  400/1000 W: 1.876, b: 0.281 Cost: 0.011366
        # Epoch  500/1000 W: 1.903, b: 0.221 Cost: 0.007024
        # Epoch  600/1000 W: 1.924, b: 0.174 Cost: 0.004340
        # Epoch  700/1000 W: 1.940, b: 0.136 Cost: 0.002682
        # Epoch  800/1000 W: 1.953, b: 0.107 Cost: 0.001657
        # Epoch  900/1000 W: 1.963, b: 0.084 Cost: 0.001024
        # Epoch 1000/1000 W: 1.971, b: 0.066 Cost: 0.000633