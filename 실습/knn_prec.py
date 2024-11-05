import math
from collections import Counter

# 두 점 사이의 유클리드 거리 계산 함수
def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2  # 각 차원의 거리 제곱을 누적
    return math.sqrt(distance)  # 제곱근을 취해 유클리드 거리 반환

# KNN 알고리즘에서 주어진 테스트 포인트와 가까운 K개의 이웃을 찾는 함수
def get_neighbors(train_data, test_point, k):
    distances = []

    for train_point in train_data:
        distance = euclidean_distance(test_point, train_point[:-1])  # 학습 데이터와의 거리 계산
        distances.append((train_point, distance))  # (데이터, 거리) 튜플로 리스트에 추가

    distances.sort(key=lambda x: x[1])  # 거리 기준으로 정렬

    neighbors = []

    for i in range(k):
        neighbors.append(distances[i][0])  # K개의 가장 가까운 이웃 선택

    return neighbors

# 가장 가까운 이웃의 클래스를 기반으로 예측 클래스 결정 함수
def predict_classification(neighbors):
    output_values = [neighbor[-1] for neighbor in neighbors]  # 이웃들의 클래스 수집
    prediction = Counter(output_values).most_common(1)[0][0]  # 가장 빈도가 높은 클래스 선택
    return prediction

# KNN 함수, 학습 데이터와 테스트 포인트, K값을 받아서 예측 결과 반환
def knn(train_data, test_point, k):
    neighbors = get_neighbors(train_data, test_point, k)  # K개의 이웃 찾기
    prediction = predict_classification(neighbors)  # 예측 클래스 결정
    return prediction

# 메인 함수
if __name__ == '__main__':
    # 학습 데이터 정의
    train_data = [
        [2.7, 2.5, 'A'],
        [1.0, 1.0, 'B'],
        [3.0, 3.5, 'A'],
        [0.5, 1.0, 'B'],
        [2.8, 2.9, 'A'],
        [0.6, 0.7, 'B']
    ]

    # 테스트 포인트 정의
    test_point = [1.5, 1.5]

    # K 값 설정
    k = 3

    # 예측 수행 및 결과 출력
    prediction = knn(train_data, test_point, k)
    print(f'The predicted class for test point {test_point} is {prediction}')
