import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import visdom

# Visdom 설정
vis = visdom.Visdom()
vis.close(env="main")

# 손실 추적 함수 정의
def loss_tracker(loss_plot, loss_value, num):
    '''손실값을 시각화'''
    vis.line(X=num,
             Y=loss_value,
             win=loss_plot,
             update='append'
             )

# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 시드 고정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 데이터 전처리 정의
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR-10 학습 데이터셋 로드
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=0)

# CIFAR-10 테스트 데이터셋 로드
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

# 클래스 라벨 정의
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 이미지 시각화를 위한 matplotlib 설정
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

# 이미지를 보여주는 함수 정의
def imshow(img):
    img = img / 2 + 0.5  # 정규화를 되돌림
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 랜덤 학습 이미지 가져오기 및 시각화
dataiter = iter(trainloader)
images, labels = next(dataiter)
vis.images(images / 2 + 0.5)  # 정규화된 이미지를 되돌림

# show images
#imshow(torchvision.utils.make_grid(images))

# 이미지와 라벨 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# truck   dog horse truck

# VGG16 모델 구현
import torchvision.models.vgg as vgg
# import vgg

cfg = [32, 32, 'M', 64, 64, 128, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M']  # VGG16 구조
# 32x32 -> 16 -> 8 -> 4

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # VGG16의 완전 연결층 정의
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),  # CIFAR-10 이미지 크기에 맞춰서 조정
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # 합성곱 계층
        x = x.view(x.size(0), -1)  # 일렬로 펼침 (평탄화)
        x = self.classifier(x)  # 완전 연결층
        return x

    # 가중치 초기화
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# VGG16 모델 인스턴스 생성 및 장치로 이동
vgg16 = VGG(vgg.make_layers(cfg), 10, True).to(device)
a = torch.Tensor(1, 3, 32, 32).to(device)
out = vgg16(a)
print(out)
# tensor([[ 3.5377e+34,  6.0071e+34, -2.7727e+34,  2.0572e+35,  2.3735e+35,
#           2.2759e+35,  5.4568e+33, -1.1127e+35,  1.0189e+35,  3.9697e+34]],
#        grad_fn=<AddmmBackward>)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.005, momentum=0.9)

# 학습률 조정 스케줄러
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# 손실 그래프 정의
loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))

# 학습 시작
epochs = 50
for epoch in range(epochs):  # 데이터셋을 여러 번 반복
    running_loss = 0.0
    lr_sche.step()
    for i, data in enumerate(trainloader, 0):
        # 입력 데이터 가져오기
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 경사 초기화
        optimizer.zero_grad()

        # 순전파, 역전파, 최적화
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 손실 값 누적
        running_loss += loss.item()
        if i % 30 == 29:  # 30 미니 배치마다 출력
            loss_tracker(loss_plt, torch.Tensor([running_loss / 30]), torch.Tensor([i + epoch * len(trainloader)]))
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 30))
            running_loss = 0.0

print('Finished Training')
# [1,    30] loss: 2.302
# [1,    60] loss: 2.297
# [1,    90] loss: 2.288
# [2,    30] loss: 2.250
# [2,    60] loss: 2.290
# ...
# [49,    60] loss: 0.075
# [49,    90] loss: 0.082
# [50,    30] loss: 0.065
# [50,    60] loss: 0.064
# [50,    90] loss: 0.060

# 테스트 데이터셋에서 일부 이미지를 가져옴
dataiter = iter(testloader)
images, labels = next(dataiter)

# 테스트 이미지 출력
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 예측 수행
outputs = vgg16(images.to(device))
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# 전체 테스트셋에서 정확도 계산
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = vgg16(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# GroundTruth:    cat  ship  ship plane
# Predicted:    cat  ship  ship plane
# Accuracy of the network on the 10000 test images: 72 %