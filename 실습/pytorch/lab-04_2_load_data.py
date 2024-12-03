from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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

class CustomDataset(Dataset):
  def __init__(self):
    self.x_data=[
      [73,80,75],
      [93,88,93],
      [89,91,90],
      [96,98,100],
      [73,66,70]]
    self.y_data=[[152],[185],[180],[196],[142]]

  def __len__(self):
    return len(self.x_data)
  
  def __getitem__(self, index):
    x = torch.FloatTensor(self.x_data[index])
    y = torch.FloatTensor(self.y_data[index])

    return x,y

# 데이터 셋
dataset=CustomDataset() 

# 데이터 로더
dataloader=DataLoader(
  dataset,
  batch_size=2, # 2의 배수
  shuffle=True,
)

model = MultivariateLinearRegressionModel()

optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs=20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
      x_train, y_train = samples
      # H(x) 계산
      prediction = model(x_train)

      # cost 계산
      cost = F.mse_loss(prediction, y_train)

      # cost로 H(x) 개선
      optimizer.zero_grad()
      cost.backward()
      optimizer.step()

      # 로그 출력
      print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
         epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item() 
      ))
      # Epoch    0/20 Batch 1/3 Cost: 35424.417969
      # Epoch    0/20 Batch 2/3 Cost: 7990.055664
      # Epoch    0/20 Batch 3/3 Cost: 2148.731934
      # Epoch    1/20 Batch 1/3 Cost: 986.442322
      # Epoch    1/20 Batch 2/3 Cost: 465.397614
      # Epoch    1/20 Batch 3/3 Cost: 88.113068
      # Epoch    2/20 Batch 1/3 Cost: 36.668221
      # Epoch    2/20 Batch 2/3 Cost: 9.448990
      # Epoch    2/20 Batch 3/3 Cost: 3.667141
      # Epoch    3/20 Batch 1/3 Cost: 0.975254
      # Epoch    3/20 Batch 2/3 Cost: 0.906193
      # Epoch    3/20 Batch 3/3 Cost: 0.006388
      # Epoch    4/20 Batch 1/3 Cost: 0.051697
      # Epoch    4/20 Batch 2/3 Cost: 0.054978
      # Epoch    4/20 Batch 3/3 Cost: 1.199195
      # Epoch    5/20 Batch 1/3 Cost: 0.401478
      # Epoch    5/20 Batch 2/3 Cost: 0.046481
      # Epoch    5/20 Batch 3/3 Cost: 0.388473
      # Epoch    6/20 Batch 1/3 Cost: 0.538637
      # Epoch    6/20 Batch 2/3 Cost: 0.145426
      # Epoch    6/20 Batch 3/3 Cost: 0.053031
      # Epoch    7/20 Batch 1/3 Cost: 0.601826
      # Epoch    7/20 Batch 2/3 Cost: 0.188046
      # Epoch    7/20 Batch 3/3 Cost: 0.048246
      # Epoch    8/20 Batch 1/3 Cost: 0.044538
      # Epoch    8/20 Batch 2/3 Cost: 0.509270
      # Epoch    8/20 Batch 3/3 Cost: 0.279848
      # Epoch    9/20 Batch 1/3 Cost: 0.051684
      # Epoch    9/20 Batch 2/3 Cost: 0.079211
      # Epoch    9/20 Batch 3/3 Cost: 1.242455
      # Epoch   10/20 Batch 1/3 Cost: 0.039547
      # Epoch   10/20 Batch 2/3 Cost: 0.284169
      # Epoch   10/20 Batch 3/3 Cost: 1.030069
      # Epoch   11/20 Batch 1/3 Cost: 0.402872
      # Epoch   11/20 Batch 2/3 Cost: 0.285181
      # Epoch   11/20 Batch 3/3 Cost: 0.017461
      # Epoch   12/20 Batch 1/3 Cost: 0.045682
      # Epoch   12/20 Batch 2/3 Cost: 0.093299
      # Epoch   12/20 Batch 3/3 Cost: 1.044303
      # Epoch   13/20 Batch 1/3 Cost: 0.401862
      # Epoch   13/20 Batch 2/3 Cost: 0.055938
      # Epoch   13/20 Batch 3/3 Cost: 0.412103
      # Epoch   14/20 Batch 1/3 Cost: 0.537061
      # Epoch   14/20 Batch 2/3 Cost: 0.010415
      # Epoch   14/20 Batch 3/3 Cost: 0.253107
      # Epoch   15/20 Batch 1/3 Cost: 0.625113
      # Epoch   15/20 Batch 2/3 Cost: 0.015669
      # Epoch   15/20 Batch 3/3 Cost: 0.070435
      # Epoch   16/20 Batch 1/3 Cost: 0.551100
      # Epoch   16/20 Batch 2/3 Cost: 0.126779
      # Epoch   16/20 Batch 3/3 Cost: 0.001243
      # Epoch   17/20 Batch 1/3 Cost: 0.074117
      # Epoch   17/20 Batch 2/3 Cost: 0.049969
      # Epoch   17/20 Batch 3/3 Cost: 1.040596
      # Epoch   18/20 Batch 1/3 Cost: 0.064074
      # Epoch   18/20 Batch 2/3 Cost: 0.334032
      # Epoch   18/20 Batch 3/3 Cost: 0.990896
      # Epoch   19/20 Batch 1/3 Cost: 0.354361
      # Epoch   19/20 Batch 2/3 Cost: 0.372612
      # Epoch   19/20 Batch 3/3 Cost: 0.308847
      # Epoch   20/20 Batch 1/3 Cost: 0.041474
      # Epoch   20/20 Batch 2/3 Cost: 0.466467
      # Epoch   20/20 Batch 3/3 Cost: 0.352104