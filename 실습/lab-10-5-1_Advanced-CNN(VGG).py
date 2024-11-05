import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

# ImageNet 챌린지 데이터셋을 기반으로 사전 학습된 모델 URL들
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()

        # Convolution
        self.features = features 
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # FC Layer
        self.classifier = nn.Sequential(
            # 이미지 사이즈에 따라 수정해야 함
            nn.Linear(512 * 7 * 7, 4096),
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
        x = self.features(x) # Convolution
        x = self.avgpool(x)  # avgpool
        x = x.view(x.size(0), -1) # 일렬로 펼침 (평탄화)
        x = self.classifier(x) # FC layer
        return x

    def _initialize_weights(self):
        # features의 값
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # activation function에 따라 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    # 1. 빈 레이어 생성, input channel = 3
    layers = []
    in_channels = 3
    
    # 2. cfg 에서 v 반복
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            # 채널 수
            in_channels = v
                     
    return nn.Sequential(*layers)
    
# 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
# 빈 레이어 생성, input channel = 3
# 1. v = 64
# conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1)
# layers += [conv2d, nn.ReLU(inplace=True)]
# in_channels = 64
# 2. v = 'M'
# layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
# 3. v = 128
# conv2d = nn.Conv2d(64, 128, kernel_size=3, padding=1)
# layers += [conv2d, nn.ReLU(inplace=True)]
# in_channels = 128
# 4. v = 'M'
# layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
# 5. v = 256
# conv2d = nn.Conv2d(128, 256, kernel_size=3, padding=1)
# layers += [conv2d, nn.ReLU(inplace=True)]
# in_channels = 256
# 6. v = 256
# conv2d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
# layers += [conv2d, nn.ReLU(inplace=True)]
# in_channels = 256
# ...


# VGG 모델 설정 (conv + fc)
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # vgg11: 8 + 3 = 11 계층
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # vgg13: 10 + 3 = 13 계층
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], # vgg16: 13 + 3 = 16 계층
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # vgg19: 16 + 3 = 19 계층
    'custom' : [64,64,64,'M',128,128,128,'M',256,256,256,'M'] # 사용자 정의 구성
}

# 사용자 정의 네트워크 구성으로 VGG 모델 생성
conv = make_layers(cfg['custom'], batch_norm=True)
CNN = VGG(make_layers(cfg['custom']), num_classes=10, init_weights=True)
print(CNN)
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace)
#     (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (5): ReLU(inplace)
#     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace)
#     (9): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (10): ReLU(inplace)
#     (11): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (12): ReLU(inplace)
#     (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace)
#     (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (17): ReLU(inplace)
#     (18): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (19): ReLU(inplace)
#     (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace)
#     (2): Dropout(p=0.5)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace)
#     (5): Dropout(p=0.5)
#     (6): Linear(in_features=4096, out_features=10, bias=True)
#   )
# )