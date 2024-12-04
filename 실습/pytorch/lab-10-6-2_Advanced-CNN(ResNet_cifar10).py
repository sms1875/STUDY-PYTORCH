# 필요한 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import visdom
import torchvision.models.resnet as resnet

# Visdom 초기화
vis = visdom.Visdom()
vis.close(env="main")

# Value Tracker 함수 정의
def value_tracker(value_plot, value, num):
    '''num, loss_value are Tensors'''
    vis.line(X=num,
             Y=value,
             win=value_plot,
             update='append')

# 장치 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (f"device is {device}")

# 랜덤 시드 설정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 데이터셋 전처리 및 정규화
transform = transforms.Compose([
    transforms.ToTensor()
])

# CIFAR10 데이터셋 로드 및 평균/표준편차 계산
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
train_data_mean = trainset.data.mean(axis=(0, 1, 2)) / 255
train_data_std = trainset.data.std(axis=(0, 1, 2)) / 255

# 데이터셋 변환 설정
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])

# 데이터셋 준비
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ResNet 모델 정의
conv1x1 = resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock = resnet.BasicBlock

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# ResNet50 모델 생성
resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], 10, True).to(device)

# 학습 준비
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(resnet50.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Loss 및 Accuracy 플롯 초기화
loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='Loss Tracker', legend=['Loss'], showlegend=True))
acc_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='Accuracy', legend=['Accuracy'], showlegend=True))

# Accuracy 체크 함수 정의
def acc_check(net, test_set, epoch, save=1):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {acc:.2f}%')
    if save:
        torch.save(net.state_dict(), f"./model/model_epoch_{epoch}_acc_{int(acc)}.pth")
    return acc

# 모델 학습
epochs = 150
for epoch in range(epochs):
    running_loss = 0.0
    # lr_sche.step()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 30 == 29:
            value_tracker(loss_plt, torch.Tensor([running_loss / 30]), torch.Tensor([i + epoch * len(trainloader)]))
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 30:.3f}')
            running_loss = 0.0
            
    lr_sche.step()
    acc = acc_check(resnet50, testloader, epoch, save=1)
    value_tracker(acc_plt, torch.Tensor([acc]), torch.Tensor([epoch]))

print('Finished Training')

# 최종 모델 평가
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = resnet50(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
