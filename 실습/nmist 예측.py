import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random
from PIL import Image

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

random.seed(1)

# parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 32 # 2의 배수



# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

# nn layers
linear1 = torch.nn.Linear(784, 256, bias=True)
linear2 = torch.nn.Linear(256, 256, bias=True)
linear3 = torch.nn.Linear(256, 10, bias=True) # 레이어 추가
relu = torch.nn.ReLU()

# xavier initialization
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)

# model
model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3)

# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss()   # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(data_loader)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
# Epoch: 0001 cost = 0.219785795
# Epoch: 0002 cost = 0.086978219
# Epoch: 0003 cost = 0.057407402
# Epoch: 0004 cost = 0.043522421
# Epoch: 0005 cost = 0.033024356
# Epoch: 0006 cost = 0.026565807
# Epoch: 0007 cost = 0.023748280
# Epoch: 0008 cost = 0.018176639
# Epoch: 0009 cost = 0.017324938
# Epoch: 0010 cost = 0.014029357
# Epoch: 0011 cost = 0.014449440
# Epoch: 0012 cost = 0.012244913
# Epoch: 0013 cost = 0.012572370
# Epoch: 0014 cost = 0.010077718
# Epoch: 0015 cost = 0.008860561

# Test the model using test sets
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float()
    Y_test = mnist_test.test_labels

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # Get one and predict
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float()
    Y_single_data = mnist_test.test_labels[r:r + 1]

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())
# Accuracy: 0.98089998960495
# Label:  4
# Prediction:  4


image_paths = [
    '/workspace/mnist 예측/0.png','/workspace/mnist 예측/1.png','/workspace/mnist 예측/2.png','/workspace/mnist 예측/3.png','/workspace/mnist 예측/4.png','/workspace/mnist 예측/5.png','/workspace/mnist 예측/6.png','/workspace/mnist 예측/7.png','/workspace/mnist 예측/8.png','/workspace/mnist 예측/9.png',
]
image_results = [0,1,2,3,4,5,6,7,8,9]

def evaluate_digit(model, image_path):
    # 이미지를 열고 흑백으로 변환
    image = Image.open(image_path).convert('L')
    
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 이미지 변환 및 모델 입력 형태로 변형
    image_tensor = transform(image).view(-1, 28 * 28).to(DEVICE)  # DEVICE로 이동
    
    # 예측
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, 1).item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0]
        confidence_score = confidence[prediction].item() * 100
        
    return prediction, confidence_score


# 결과 저장 및 출력
results = []
for path in image_paths:
    pred, conf = evaluate_digit(model, path)
    results.append({
        'image': path,
        'prediction': pred,
        'confidence': conf
    })

for result in results:
    print(result)