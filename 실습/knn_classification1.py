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

    # 학습 데이터와 테스트 포인트 사이의 거리 계산
    for train_point in train_data:
        distance = euclidean_distance(test_point, train_point[:-1])  # 마지막 열(클래스 레이블)을 제외하고 거리 계산
        distances.append((train_point, distance))  # (데이터, 거리) 튜플로 리스트에 추가

    distances.sort(key=lambda x: x[1])  # 거리 기준으로 오름차순 정렬

    neighbors = []

    # K개의 가장 가까운 이웃 선택
    for i in range(k):
        neighbors.append(distances[i][0])  # 거리 순으로 K개의 이웃 추가

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
    # 학습 데이터 정의 (마지막 열이 클래스 레이블)
    train_data = [
        [300, 150, 50, 1.2, 20, 'A'],
        [310, 145, 55, 1.3, 22, 'A'],
        [290, 152, 48, 1.1, 19, 'A'],
        [250, 100, 60, 0.8, 12, 'B'],
        [255, 98, 62, 0.85, 11, 'B'],
        [260, 105, 59, 0.9, 13, 'B'],
        [320, 160, 45, 1.5, 25, 'A'],
        [245, 90, 63, 0.75, 10, 'B'],
        [280, 130, 70, 1.0, 15, 'C'],
        [285, 135, 68, 1.05, 16, 'C'],
        [275, 120, 65, 1.0, 14, 'C'],
        [295, 140, 72, 1.2, 17, 'C'],
    ]

    # 테스트 포인트 정의 (클래스 레이블 없음)
    test_points = [
        [300, 140, 60, 1.1, 18],
        [270, 125, 66, 0.95, 14],
        [310, 155, 53, 1.4, 23]
    ]

    # K 값 설정
    k = 3

    # 각 테스트 포인트에 대해 예측 수행 및 결과 출력
    for i in range(len(test_points)):
        prediction = knn(train_data, test_points[i], k)  # 테스트 포인트에 대한 예측 수행
        print(f'The predicted class for test point {test_points[i]} is {prediction}')
