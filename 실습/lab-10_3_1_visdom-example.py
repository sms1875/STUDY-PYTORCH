import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as dsets

import visdom
vis = visdom.Visdom()

# 텍스트
vis.text("Hello, world!",env="main")

# 이미지 생성
a=torch.randn(3,200,200)
vis.image(a)

# 이미지 여러개 생성
vis.images(torch.Tensor(3,3,28,28))

# using MNIST and CIFAR10
MNIST = dsets.MNIST(root="./MNIST_data",train = True,transform=torchvision.transforms.ToTensor(), download=True)
cifar10 = dsets.CIFAR10(root="./cifar10",train = True, transform=torchvision.transforms.ToTensor(),download=True)

# CIFAR10
data = cifar10.__getitem__(0)
print(data[0].shape) 
# torch.Size([3, 32, 32])
vis.images(data[0],env="main")

# MNIST
data = MNIST.__getitem__(0)
print(data[0].shape)
# torch.Size([1, 28, 28])
vis.images(data[0],env="main")

# Check dataset
data_loader = torch.utils.data.DataLoader(dataset = MNIST,
                                          batch_size = 32,
                                          shuffle = False)
for num, value in enumerate(data_loader):
    value = value[0]
    print(value.shape)
    # torch.Size([32, 1, 28, 28])
    vis.images(value)
    break

# Line Plot
Y_data = torch.randn(5)
plt = vis.line (Y=Y_data)

X_data = torch.Tensor([1,2,3,4,5])
plt = vis.line(Y=Y_data, X=X_data)

# Line update
Y_append = torch.randn(1)
X_append = torch.Tensor([6])

vis.line(Y=Y_append, X=X_append, win=plt, update='append')

# multiple Line on single windows
num = torch.Tensor(list(range(0,10)))
num = num.view(-1,1)
num = torch.cat((num,num),dim=1)

plt = vis.line(Y=torch.randn(10,2), X = num)
plt2 = vis.line(Y=torch.randn(10,2), X = num)

# Line info
plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', showlegend=True))
plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', legend = ['1번'],showlegend=True))
plt = vis.line(Y=torch.randn(10,2), X = num, opts=dict(title='Test', legend=['1번','2번'],showlegend=True))

# make function for update line
def loss_tracker(loss_plot, loss_value, num):
    '''num, loss_value, are Tensor'''
    vis.line(X=num,
             Y=loss_value,
             win = loss_plot,
             update='append'
             )
plt = vis.line(Y=torch.Tensor(1).zero_())

for i in range(500):
    loss = torch.randn(1) + i
    loss_tracker(plt, loss, torch.Tensor([i]))

# 종료
vis.close(env="main")