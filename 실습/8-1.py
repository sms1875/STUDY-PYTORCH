import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# XOR 
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# nn layers
linear = torch.nn.Linear(2, 1, bias=True) #layer
sigmoid = torch.nn.Sigmoid() #활성화 함수

# model
model = torch.nn.Sequential(linear, sigmoid).to(device) 

# define cost/loss & optimizer
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

#학습
for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    # cost/loss function
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, cost.item())
        
# 0 0.7273974418640137
# 100 0.6931476593017578
# 200 0.6931471824645996
# 300 0.6931471824645996
# ...
# 9800 0.6931471824645996
# 9900 0.6931471824645996
# 10000 0.6931471824645996
        
        
# Accuracy computation
# True if hypothesis>0.5 else False
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('\nHypothesis: ', hypothesis.detach().cpu().numpy(), '\nCorrect: ', predicted.detach().cpu().numpy(), '\nAccuracy: ', accuracy.item())

# Hypothesis:  [[0.5]
#  [0.5]
#  [0.5]
#  [0.5]] 
# Correct:  [[0.]
#  [0.]
#  [0.]
#  [0.]] 
# Accuracy:  0.5