import torch
import torch.optim as optim
import numpy as np
torch.manual_seed(0)

# 문자 집합 정의
char_set = ['h', 'i', 'e', 'l', 'o']

# 하이퍼파라미터 설정
input_size = len(char_set)  # 입력의 크기 (문자 집합의 크기)
hidden_size = len(char_set)  # 은닉 상태의 크기
learning_rate = 0.1

# 데이터 설정
# 입력 데이터: 'hihell'을 인덱스로 표현
x_data = [[0, 1, 0, 2, 3, 3]]
# 입력 데이터의 원-핫 인코딩 표현
x_one_hot = [[[1, 0, 0, 0, 0],  # 'h'
              [0, 1, 0, 0, 0],  # 'i'
              [1, 0, 0, 0, 0],  # 'h'
              [0, 0, 1, 0, 0],  # 'e'
              [0, 0, 0, 1, 0],  # 'l'
              [0, 0, 0, 1, 0]]]  # 'l'
# 출력 데이터: 'ihello'를 인덱스로 표현
y_data = [[1, 0, 2, 3, 3, 4]]

# 데이터를 PyTorch 텐서로 변환
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data) 

# RNN 모델 선언
# batch_first=True로 설정하여 입력 데이터의 첫 번째 차원이 배치 크기임을 명시
rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True) # batch_first guarantees the order of output = (B, S, F)

criterion = torch.nn.CrossEntropyLoss() 
optimizer = optim.Adam(rnn.parameters(), learning_rate)  

# 학습 시작
for i in range(100):
    optimizer.zero_grad()  
    outputs, _status = rnn(X)  # RNN에 입력 데이터 전달하여 출력과 상태 반환

    # 출력 텐서를 2차원으로 변환하여 손실 계산
    loss = criterion(outputs.view(-1, input_size), Y.view(-1))
    loss.backward() 
    optimizer.step() 

    # 예측 결과를 numpy 배열로 변환하여 최종 예측 값 도출
    result = outputs.data.numpy().argmax(axis=2)
    # 예측 인덱스를 문자로 변환하여 문자열 생성
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)

# 0 loss:  1.7802648544311523 prediction:  [[1 1 1 1 1 1]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  iiiiii
# 1 loss:  1.4931949377059937 prediction:  [[1 4 1 1 4 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ioiioo
# 2 loss:  1.3337111473083496 prediction:  [[1 3 2 3 1 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ilelio
# 3 loss:  1.215294599533081 prediction:  [[2 3 2 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  elelll
# 4 loss:  1.1131387948989868 prediction:  [[2 3 2 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  elelll
# ...
# 96 loss:  0.5322802662849426 prediction:  [[1 3 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ilello
# 97 loss:  0.5321123600006104 prediction:  [[1 3 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ilello
# 98 loss:  0.5319532752037048 prediction:  [[1 3 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ilello
# 99 loss:  0.5317899584770203 prediction:  [[1 3 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] prediction str:  ilello
