import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib.pyplot import imshow
import os

# 파일을 저장할 경로 생성
os.makedirs('custom_data/train_data/gray', exist_ok=True)
os.makedirs('custom_data/train_data/red', exist_ok=True)

# 이미지 전처리를 위한 변환(transform) 정의
trans = transforms.Compose([
    transforms.Resize((64, 128))  # 이미지를 (64, 128) 크기로 리사이즈
])

# ImageFolder를 이용해 데이터셋 로드
# 'root' 디렉토리의 하위 폴더를 클래스 레이블로 매핑
train_data = torchvision.datasets.ImageFolder(
    root='custom_data/origin_data',  # 원본 이미지 경로
    transform=trans  # 정의된 변환 적용
)

# 데이터셋 순회 및 저장
for num, value in enumerate(train_data):
    # 데이터와 레이블 분리
    data, label = value
    print(num, data, label)  # 현재 데이터의 순번, 이미지, 레이블 출력

    # 레이블에 따라 이미지를 다른 폴더에 저장
    if label == 0:
        # 레이블이 0인 이미지를 'gray' 폴더에 저장
        data.save('custom_data/train_data/gray/%d_%d.jpeg' % (num, label))
    else:
        # 레이블이 0이 아닌 이미지를 'red' 폴더에 저장
        data.save('custom_data/train_data/red/%d_%d.jpeg' % (num, label))
