import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

def sigmoid(x):
    #  sigmoid function
    return 1.0 / (1.0 + torch.exp(-x))
  
# sigmoid를 미분하는 함수
def sigmoid_prime(x):
    # derivative of the sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))

# XOR 데이터셋
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device) 
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# nn layers (직접 w, b 선언)
w1 = torch.randn(2, 2).to(device)  # 입력 2, 출력 2
b1 = torch.randn(2).to(device)      # 2개의 출력
w2 = torch.randn(2, 1).to(device)   # 입력 2, 출력 1
b2 = torch.randn(1).to(device)      # 1개의 출력

lr = 1

for step in range(10001):
    # forward
    l1 = torch.add(torch.matmul(X, w1), b1)  # w*x + b
    a1 = sigmoid(l1)  # 활성화 함수(activation function)
    l2 = torch.add(torch.matmul(a1, w2), b2)
    Y_pred = sigmoid(l2)  # 예측값
    
    # binary cross entropy loss 사용 
    cost = -torch.mean(Y * torch.log(Y_pred) + (1 - Y) * torch.log(1 - Y_pred)) 
    
    # Back prop (chain rule)
    d_Y_pred = (Y_pred - Y) / (Y_pred * (1.0 - Y_pred) + 1e-7)
    
    # Layer 2    
    d_l2 = d_Y_pred * sigmoid_prime(l2)  
    d_b2 = d_l2  # bias 미분
    d_w2 = torch.matmul(torch.transpose(a1, 0, 1), d_b2)  # weight 미분
    
    # Layer 1
    d_a1 = torch.matmul(d_b2, torch.transpose(w2, 0, 1))  
    d_l1 = d_a1 * sigmoid_prime(l1)  
    d_b1 = d_l1
    d_w1 = torch.matmul(torch.transpose(X, 0, 1), d_b1)  # weight 미분
    
    # Weight update
    w1 = w1 - lr * d_w1
    b1 = b1 - lr * torch.mean(d_b1, 0)
    w2 = w2 - lr * d_w2
    b2 = b2 - lr * torch.mean(d_b2, 0)

    if step % 100 == 0:
        print(step, cost.item())
# 0 0.8633197546005249
# 100 0.17553459107875824
# 200 0.03060678020119667
# 300 0.015832239761948586
# ...
# 9800 0.0003155834274366498
# 9900 0.000312378368107602
# 10000 0.00030918820993974805