import numpy as np
import torch

# NumPy 복습
# 1D Array with NumPy
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)
# [0. 1. 2. 3. 4. 5. 6.]
# 배열의 차원과 형태 출력
print('Rank  of t: ', t.ndim)
print('Shape of t: ', t.shape)
# Rank  of t:  1
# Shape of t:  (7,)

# 배열의 요소 접근 및 슬라이싱 예제
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1])  # 요소 접근
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1])    # 슬라이싱
print('t[:2] t[3:]     = ', t[:2], t[3:])       # 슬라이싱
# t[0] t[1] t[-1] =  0.0 1.0 6.0
# t[2:5] t[4:-1]  =  [2. 3. 4.] [4. 5.]
# t[:2] t[3:]     =  [0. 1.] [3. 4. 5. 6.]

# 2D Array with NumPy
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
# [[ 1.  2.  3.]
#  [ 4.  5.  6.]
#  [ 7.  8.  9.]
#  [10. 11. 12.]]
# 2차원 배열의 차원과 형태 출력
print('Rank  of t: ', t.ndim)
print('Shape of t: ', t.shape)
# Rank  of t:  2
# Shape of t:  (4, 3)

# PyTorch는 NumPy와 유사함 (하지만 더 강력함)
# 1D Array with PyTorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
# tensor([0., 1., 2., 3., 4., 5., 6.])

# 텐서의 차원(rank)과 형태(shape) 출력
print(t.dim())  # 차원
print(t.shape)  # 형태
print(t.size()) # 형태
print(t[0], t[1], t[-1])  # 요소 접근
print(t[2:5], t[4:-1])    # 슬라이싱
print(t[:2], t[3:])       # 슬라이싱
# 1
# torch.Size([7])
# torch.Size([7])
# tensor(0.) tensor(1.) tensor(6.)
# tensor([2., 3., 4.]) tensor([4., 5.])
# tensor([0., 1.]) tensor([3., 4., 5., 6.])

# 2D Array with PyTorch
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])
print(t)
# tensor([[ 1.,  2.,  3.],
#         [ 4.,  5.,  6.],
#         [ 7.,  8.,  9.],
#         [10., 11., 12.]])

# 2차원 텐서의 차원과 형태 출력
print(t.dim())  # 차원
print(t.size()) # 형태
print(t[:, 1])  # 특정 열 추출
print(t[:, 1].size())  # 특정 열의 형태
print(t[:, :-1])       # 특정 열 제외한 나머지 추출
# 2
# torch.Size([4, 3])
# tensor([ 2.,  5.,  8., 11.])
# torch.Size([4])
# tensor([[ 1.,  2.],
#         [ 4.,  5.],
#         [ 7.,  8.],
#         [10., 11.]])

# Shape, Rank, Axis 예제
t = torch.FloatTensor([[[[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12]],
                       [[13, 14, 15, 16],
                        [17, 18, 19, 20],
                        [21, 22, 23, 24]]
                       ]])
print(t.dim())  # 차원(rank) = 4
print(t.size()) # 형태(shape) = (1, 2, 3, 4)
# 4
# torch.Size([1, 2, 3, 4])

# PyTorch에서 자주 사용되는 연산들
# Mul vs. Matmul
print()
print('-------------')
print('Mul vs Matmul')
print('-------------')
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
# Shape of Matrix 1:  torch.Size([2, 2])
# Shape of Matrix 2:  torch.Size([2, 1])

print(m1.matmul(m2)) # 행렬 곱 (2 x 1)
# tensor([[ 5.],
#         [11.]])

print(m1 * m2) # 원소별 곱
print(m1.mul(m2))
# tensor([[1., 2.],
#         [6., 8.]])
# tensor([[1., 2.],
#         [6., 8.]])

# Broadcasting
# 동일한 형태
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)  # (1, 2) + (1, 2) -> (1, 2)
# tensor([[5., 5.]])

# 벡터 + 스칼라
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # 3 -> [[3, 3]]
print(m1 + m2)
# tensor([[4., 5.]])

# 2 x 1 벡터 + 1 x 2 벡터
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)
# tensor([[4., 5.],
#         [5., 6.]])

# 평균(mean) 구하기
t = torch.FloatTensor([1, 2])
print(t.mean())  # 텐서의 전체 평균 계산
# tensor(1.5000)

# 정수형 텐서에서 평균을 구할 수 없음
t = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)
# Can only calculate the mean of floating types. Got Long instead.

# 고차원 텐서에서 특정 차원을 따라 평균 계산
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.mean())         # 전체 평균
print(t.mean(dim=0))    # 행 기준 평균
print(t.mean(dim=1))    # 열 기준 평균
print(t.mean(dim=-1))   # 마지막 차원 기준 평균
# tensor([[1., 2.],
#         [3., 4.]])
# tensor(2.5000)
# tensor([2., 3.])
# tensor([1.5000, 3.5000])
# tensor([1.5000, 3.5000])

# 합(sum) 구하기
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.sum())          # 전체 합
print(t.sum(dim=0))     # 행 기준 합
print(t.sum(dim=1))     # 열 기준 합
print(t.sum(dim=-1))    # 마지막 차원 기준 합
# tensor([[1., 2.],
#         [3., 4.]])
# tensor(10.)
# tensor([4., 6.])
# tensor([3., 7.])
# tensor([3., 7.])

# Max와 Argmax
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.max())          # 전체에서 최댓값 반환
print(t.max(dim=0))     # 각 차원의 최댓값과 해당 인덱스 반환
print('Max: ', t.max(dim=0)[0])     # 최댓값
print('Argmax: ', t.max(dim=0)[1])  # 인덱스
print(t.max(dim=1))     # 행 기준 최댓값
print(t.max(dim=-1))    # 마지막 차원 기준 최댓값
# tensor([[1., 2.],
#         [3., 4.]])
# tensor(4.)
# (tensor([3., 4.]), tensor([1, 1]))
# Max:  tensor([3., 4.])
# Argmax:  tensor([1, 1])
# (tensor([2., 4.]), tensor([1, 1]))
# (tensor([2., 4.]), tensor([1, 1]))

# View (텐서 형태 변경)
t = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)         # 초기 형태 출력
print(ft.view([-1, 3])) # 새로운 형태로 변경
print(ft.view([-1, 3]).shape)  # 변경 후 형태 출력
# torch.Size([2, 2, 3])
# tensor([[ 0.,  1.,  2.],
#         [ 3.,  4.,  5.],
#         [ 6.,  7.,  8.],
#         [ 9., 10., 11.]])
# torch.Size([4, 3])

# 차원 확장과 축소 (Squeeze, Unsqueeze)
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
print(ft.squeeze())  # 차원이 1인 경우 축소
print(ft.unsqueeze(0))  # 0번째 차원에 새 축 추가
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])
# tensor([0., 1., 2.])
# tensor([[[0.],
#          [1.],
#          [2.]]])

# One-hot Encoding을 위한 Scatter 사용 예제
lt = torch.LongTensor([[0], [1], [2], [0]])
one_hot = torch.zeros(4, 3) # 배치 크기 4, 클래스 3
one_hot.scatter_(1, lt, 1)  # one-hot 인코딩 적용
print(one_hot)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.],
#         [1., 0., 0.]])

# 데이터 타입 변환
lt = torch.LongTensor([1, 2, 3, 4])
print(lt.float())       # 실수형으로 변환
# tensor([1., 2., 3., 4.])

bt = torch.ByteTensor([True, False, False, True])
print(bt.long())        # 정수형으로 변환
print(bt.float())       # 실수형으로 변환
# tensor([1, 0, 0, 1])
# tensor([1., 0., 0., 1.])

# 텐서 연결 (Concatenation)
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x, y], dim=0))  # 행 기준 연결
print(torch.cat([x, y], dim=1))  # 열 기준 연결
# tensor([[1., 2.],
#         [3., 4.],
#         [5., 6.],
#         [7., 8.]])
# tensor([[1., 2., 5., 6.],
#         [3., 4., 7., 8.]])

# 텐서 쌓기 (Stacking)
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))     # 기본 쌓기
print(torch.stack([x, y, z], dim=1))  # 축을 지정하여 쌓기
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])

# Ones와 Zeros를 사용한 텐서 초기화
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(torch.ones_like(x))   # 동일한 형태의 1로 초기화된 텐서
print(torch.zeros_like(x))  # 동일한 형태의 0으로 초기화된 텐서
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

# 인플레이스 연산 (In-place Operation)
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.))  # 새 텐서를 생성하여 연산
print(x)          # 원본 텐서는 그대로
print(x.mul_(2.)) # 원본 텐서에 연산 결과를 반영
print(x)
# tensor([[2., 4.],
#         [6., 8.]])
# tensor([[1., 2.],
#         [3., 4.]])
# tensor([[2., 4.],
#         [6., 8.]])
# tensor([[2., 4.],
#         [6., 8.]])

# Zip 예제
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
# 1 4
# 2 5
# 3 6
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
# 1 4 7
# 2 5 8
# 3 6 9