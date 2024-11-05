import math
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# 두 점 사이의 유클리드 거리 계산 함수
def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# KNN 알고리즘에서 주어진 테스트 포인트와 가까운 K개의 이웃을 찾는 함수
def get_neighbors(train_data, test_point, k):
    distances = []

    for train_point in train_data:
        distance = euclidean_distance(test_point, train_point[:-1])  # 클래스 레이블 제외하고 거리 계산
        distances.append((train_point, distance))

    distances.sort(key=lambda x: x[1])  # 거리 기준으로 오름차순 정렬

    neighbors = [distances[i][0] for i in range(k)]  # K개의 가장 가까운 이웃 선택
    return neighbors

# 가장 가까운 이웃의 클래스를 기반으로 예측 클래스 결정 함수
def predict_classification(neighbors):
    output_values = [neighbor[-1] for neighbor in neighbors]  # 이웃들의 클래스 수집
    prediction = Counter(output_values).most_common(1)[0][0]  # 가장 빈도가 높은 클래스 선택
    return prediction

# KNN 함수, 학습 데이터와 테스트 포인트, K값을 받아서 예측 결과 반환
def knn(train_data, test_point, k):
    neighbors = get_neighbors(train_data, test_point, k)
    prediction = predict_classification(neighbors)
    return prediction

# 메인 함수
if __name__ == '__main__':
    # 학습 데이터 정의 (마지막 열이 클래스 레이블)
    train_data = [
        [5.2, 2.1, '오전', 3, 'A'],
        [4.8, 1.8, '오전', 2, 'A'],
        [12.5, 4.2, '저녁', 5, 'C'],
        [11.7, 4.0, '저녁', 6, 'C'],
        [3.1, 1.2, '밤', 2, 'B'],
        [2.9, 1.0, '밤', 1, 'B'],
        [6.5, 2.8, '오후', 4, 'A'],
        [13.2, 4.5, '저녁', 5, 'C'],
        [3.3, 1.1, '밤', 2, 'B'],
        [5.5, 2.3, '오전', 3, 'A'],
        [4.2, 1.6, '오후', 3, 'A'],
        [3.0, 1.2, '밤', 2, 'B'],
    ]

    test_points = [
        [4.5, 1.9, '오전', 3],
        [12.0, 4.1, '저녁', 5],
        [3.2, 1.3, '밤', 2],
    ]

    # LabelEncoder를 사용해 시간대를 숫자로 인코딩
    le = LabelEncoder()

    # 학습 데이터에서 시간대 부분 추출
    time_of_day_train = [data[2] for data in train_data]
    time_of_day_test = [point[2] for point in test_points]

    # 학습 및 테스트 데이터의 시간대 인코딩
    le.fit(time_of_day_train + time_of_day_test)
    encoded_train_time = le.transform(time_of_day_train)
    encoded_test_time = le.transform(time_of_day_test)

    # 인코딩된 값을 학습 및 테스트 데이터에 적용
    encoded_train_data = [
        train_data[i][:2] + [encoded_train_time[i]] + train_data[i][3:] for i in range(len(train_data))
    ]
    encoded_test_points = [
        test_points[i][:2] + [encoded_test_time[i]] + test_points[i][3:] for i in range(len(test_points))
    ]

    # K 값 설정
    k = 3

    # 각 테스트 포인트에 대해 예측 수행 및 결과 출력
    for test_point in encoded_test_points:
        prediction = knn(encoded_train_data, test_point, k)
        print(f'The predicted class for test point {test_point} is {prediction}')
