import torch
import numpy as np

# 결과를 재현 가능하게 만들기 위해 랜덤 시드 설정
torch.manual_seed(0)

# 입력 데이터의 크기와 히든 상태의 크기 정의
input_size = 4  # 입력 데이터의 차원
hidden_size = 2  # 히든 스테이트의 차원

# RNN에 입력될 데이터를 생성하는 부분

# 'hello'와 같은 단어를 원-핫 인코딩 방식으로 표현
h = [1, 0, 0, 0]  # 'h'를 원-핫 인코딩
e = [0, 1, 0, 0]  # 'e'를 원-핫 인코딩
l = [0, 0, 1, 0]  # 'l'을 원-핫 인코딩
o = [0, 0, 0, 1]  # 'o'를 원-핫 인코딩

# 세 개의 시퀀스 데이터 생성
# 각 시퀀스는 (5, 4) 크기를 가지며, 총 3개의 시퀀스를 포함
input_data_np = np.array([
    [h, e, l, l, o],  # 첫 번째 시퀀스 ('hello')
    [e, o, l, l, l],  # 두 번째 시퀀스
    [l, l, e, e, l]   # 세 번째 시퀀스
], dtype=np.float32)

# numpy 배열을 PyTorch 텐서로 변환
input_data = torch.Tensor(input_data_np)

# RNN 모델 선언
# 입력 크기(input_size)와 히든 크기(hidden_size)를 설정
rnn = torch.nn.RNN(input_size, hidden_size)

# RNN에 입력 데이터를 전달하여 출력 계산
outputs, _status = rnn(input_data)

print(outputs)
# tensor([[[-0.7497, -0.6135],
#          [-0.5282, -0.2473],
#          [-0.9136, -0.4269],
#          [-0.9136, -0.4269],
#          [-0.9028,  0.1180]],

#         [[-0.5753, -0.0070],
#          [-0.9052,  0.2597],
#          [-0.9173, -0.1989],
#          [-0.9173, -0.1989],
#          [-0.8996, -0.2725]],

#         [[-0.9077, -0.3205],
#          [-0.8944, -0.2902],
#          [-0.5134, -0.0288],
#          [-0.5134, -0.0288],
#          [-0.9127, -0.2222]]], grad_fn=<StackBackward0>)

print(outputs.size())  # (배치 크기, 시퀀스 길이, 히든 상태 크기)
# torch.Size([3, 5, 2])