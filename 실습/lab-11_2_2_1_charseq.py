import torch
import torch.optim as optim
import numpy as np
torch.manual_seed(0)

# 예제 문자열
sample = " if you want you"

# 문자 집합 정의
char_set = list(set(sample))  # 문자열의 중복을 제거하여 문자 집합 생성
char_dic = {c: i for i, c in enumerate(char_set)}  # 문자에 인덱스를 매핑한 딕셔너리 생성

# 하이퍼파라미터 설정
dic_size = len(char_dic)  # 입력의 크기 (문자 집합의 크기)
hidden_size = len(char_dic)  # 은닉 상태의 크기
learning_rate = 0.1

# 데이터 설정
# 입력 문자열을 인덱스로 변환
sample_idx = [char_dic[c] for c in sample]  # 문자열의 각 문자를 대응되는 인덱스로 변환
x_data = [sample_idx[:-1]]  # 입력 데이터는 마지막 문자를 제외한 부분
x_one_hot = [np.eye(dic_size)[x] for x in x_data]  # 입력 데이터를 원-핫 인코딩으로 변환
y_data = [sample_idx[1:]]  # 출력 데이터는 첫 번째 문자를 제외한 부분

# 데이터를 PyTorch 텐서로 변환
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data) 

# RNN 모델 선언
# batch_first=True로 설정하여 입력 데이터의 첫 번째 차원이 배치 크기임을 명시
rnn = torch.nn.RNN(dic_size, hidden_size, batch_first=True) # batch_first guarantees the order of output = (B, S, F)

criterion = torch.nn.CrossEntropyLoss() 
optimizer = optim.Adam(rnn.parameters(), learning_rate)  

# 학습 시작
for i in range(100):
    optimizer.zero_grad()  
    outputs, _status = rnn(X)  # RNN에 입력 데이터 전달하여 출력과 상태 반환

    # 출력 텐서를 2차원으로 변환하여 손실 계산
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward() 
    optimizer.step() 

    # 예측 결과를 numpy 배열로 변환하여 최종 예측 값 도출
    result = outputs.data.numpy().argmax(axis=2)
    # 예측 인덱스를 문자로 변환하여 문자열 생성
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)

# 0 loss:  2.342663288116455 prediction:  [[8 7 7 8 5 0 0 8 7 0 8 5 8 5 0]] true Y:  [[4, 3, 2, 0, 7, 9, 2, 5, 1, 8, 6, 2, 0, 7, 9]] prediction str:  noonwyynoynwnwy
# 1 loss:  2.005516529083252 prediction:  [[8 7 2 8 7 0 0 8 7 2 7 2 8 7 0]] true Y:  [[4, 3, 2, 0, 7, 9, 2, 5, 1, 8, 6, 2, 0, 7, 9]] prediction str:  no noyyno o noy
# 2 loss:  1.7695480585098267 prediction:  [[8 7 2 0 7 9 0 5 7 2 7 2 0 7 9]] true Y:  [[4, 3, 2, 0, 7, 9, 2, 5, 1, 8, 6, 2, 0, 7, 9]] prediction str:  no youywo o you
# 3 loss:  1.5684891939163208 prediction:  [[5 7 2 0 7 9 2 5 1 9 7 2 0 7 9]] true Y:  [[4, 3, 2, 0, 7, 9, 2, 5, 1, 8, 6, 2, 0, 7, 9]] prediction str:  wo you wauo you
# 4 loss:  1.4520589113235474 prediction:  [[5 3 2 0 7 9 2 5 1 9 6 2 0 7 9]] true Y:  [[4, 3, 2, 0, 7, 9, 2, 5, 1, 8, 6, 2, 0, 7, 9]] prediction str:  wf you waut you
# ...
# 96 loss:  0.8099259734153748 prediction:  [[4 3 2 0 7 9 2 5 1 8 6 2 0 7 9]] true Y:  [[4, 3, 2, 0, 7, 9, 2, 5, 1, 8, 6, 2, 0, 7, 9]] prediction str:  if you want you
# 97 loss:  0.8098456263542175 prediction:  [[4 3 2 0 7 9 2 5 1 8 6 2 0 7 9]] true Y:  [[4, 3, 2, 0, 7, 9, 2, 5, 1, 8, 6, 2, 0, 7, 9]] prediction str:  if you want you
# 98 loss:  0.8097667694091797 prediction:  [[4 3 2 0 7 9 2 5 1 8 6 2 0 7 9]] true Y:  [[4, 3, 2, 0, 7, 9, 2, 5, 1, 8, 6, 2, 0, 7, 9]] prediction str:  if you want you
# 99 loss:  0.8096891045570374 prediction:  [[4 3 2 0 7 9 2 5 1 8 6 2 0 7 9]] true Y:  [[4, 3, 2, 0, 7, 9, 2, 5, 1, 8, 6, 2, 0, 7, 9]] prediction str:  if you want you
