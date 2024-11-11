import pandas as pd
import numpy as np
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from visdom import Visdom
import random

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is {device}")

# 재현성을 위한 시드 설정
random.seed(777)
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

# 상수 선언
TRAIN_DATA_PATH = "dataset/train.csv"
TEST_DATA_PATH = "dataset/test.csv"
SUBMISSION_PATH = "dataset/submit.csv"

EPOCHS = 1000
LEARNING_RATE = 0.001
N_FEATURES_TO_SELECT = 9  # 선택할 특성 수

# 사용 가능한 Feature 목록
FEATURES = [
    "wind_direction",
    "sky_condition",
    "precipitation_form",
    "wind_speed",
    "humidity",
    "low_temp",
    "high_temp",
    "Precipitation_Probability",
    "day_of_week",
    "is_workday",
    "feels_like_temp",
    "avg_temp",
    "temp_humidity_interaction",
]

TARGET = "number_of_rentals"

# Visdom 초기화
viz = Visdom()
viz.close()


class FeatureSelector:
    def __init__(self, n_features=5, method="mutual_info"):
        self.n_features = n_features
        self.method = method
        self.selected_features = None
        self.selector = None

    def select_features(self, X, y):
        if self.method == "mutual_info":
            self.selector = SelectKBest(
                score_func=mutual_info_regression, k=self.n_features
            )
            self.selector.fit(X, y)
            feature_scores = pd.DataFrame(
                {"Feature": X.columns, "Score": self.selector.scores_}
            )
            self.selected_features = feature_scores.nlargest(self.n_features, "Score")[
                "Feature"
            ].tolist()
            print("\nFeature importance scores:")
            print(feature_scores.sort_values("Score", ascending=False))
        return self.selected_features


class RentalPredictionModel(nn.Module):
    def __init__(self, input_dim):
        super(RentalPredictionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


def preprocess_data(data, is_test=False):
    """데이터 전처리 함수"""
    data["date_time"] = pd.to_datetime(data["date_time"])
    data["day_of_week"] = data["date_time"].dt.dayofweek
    data["is_workday"] = data["day_of_week"].apply(lambda x: 1 if x < 5 else 0)

    # 체감 온도 계산
    data["feels_like_temp"] = (
        13.12
        + 0.6215 * data["low_temp"]
        - 11.37 * np.power(data["wind_speed"], 0.16)
        + 0.3965 * data["low_temp"] * np.power(data["wind_speed"], 0.16)
    )

    # 평균 온도 계산
    data["avg_temp"] = (data["low_temp"] + data["high_temp"]) / 2

    # 온도와 습도의 상호작용
    data["temp_humidity_interaction"] = data["avg_temp"] * data["humidity"]

    return data


def train_and_validate(X, y, features, epochs=100, learning_rate=0.001):
    """모델 학습 및 검증 함수"""
    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X[features], y, test_size=0.2, random_state=42
    )

    # 데이터 정규화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 모델 초기화
    model = RentalPredictionModel(len(features)).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 텐서 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.log1p(
        torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    ).to(device)

    # 학습 진행률 표시를 위한 설정
    print_interval = max(epochs // 10, 1)

    # 학습
    train_losses = []
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (epoch + 1) % print_interval == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 검증
    model.eval()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.log1p(
        torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    ).to(device)

    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        val_loss = criterion(y_val_pred, y_val_tensor).item()

        # 예측값과 실제값 변환 (로그 변환 복원)
        y_val_pred_exp = torch.expm1(y_val_pred).cpu().numpy()
        y_val_actual = torch.expm1(y_val_tensor).cpu().numpy()

    # 학습 과정 시각화
    viz.line(
        Y=np.array(train_losses),
        X=np.arange(len(train_losses)),
        opts=dict(title="Training Loss Over Time", xlabel="Epoch", ylabel="Loss"),
    )

    return model, scaler, val_loss, y_val_pred_exp, y_val_actual


def predict_test_data(model, X_test, scaler):
    """테스트 데이터 예측 함수"""
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        model.eval()
        y_test_pred = model(X_test_tensor)
        y_test_pred_exp = torch.expm1(y_test_pred).cpu().numpy()

    return y_test_pred_exp


def save_predictions(predictions, date_time, save_path):
    """예측 결과 저장 함수"""
    submission = pd.DataFrame(
        {"date_time": date_time, "number_of_rentals": predictions.flatten()}
    )
    submission.to_csv(save_path, index=False)
    print(f"예측 결과가 {save_path}에 저장되었습니다.")


def evaluate_predictions(y_true, y_pred):
    """예측 결과 평가 함수"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    print("\nValidation Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    return mse, rmse, mae


def main():
    print("데이터 로드 및 전처리 시작...")
    # 데이터 로드 및 전처리
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    train_data = preprocess_data(train_data)

    X = train_data[FEATURES]
    y = train_data[TARGET]

    print("\n특성 선택 시작...")
    # Feature Selection
    selector = FeatureSelector(n_features=N_FEATURES_TO_SELECT)
    optimal_features = selector.select_features(X, y)
    print(f"\n선택된 특성: {optimal_features}")

    print("\n모델 학습 시작...")
    # 선택된 특성으로 모델 학습
    model, scaler, val_loss, y_val_pred, y_val_actual = train_and_validate(
        X, y, optimal_features, epochs=EPOCHS, learning_rate=LEARNING_RATE
    )
    print(f"\n검증 손실: {val_loss:.4f}")

    # 예측 결과 평가
    evaluate_predictions(y_val_actual, y_val_pred)

    # 예측 결과 시각화
    viz.line(
        X=np.arange(len(y_val_actual)),
        Y=np.column_stack((y_val_actual, y_val_pred)),
        opts=dict(
            title="Validation: Predicted vs Actual",
            xlabel="Sample Index",
            ylabel="Number of Rentals",
            legend=["Actual", "Predicted"],
        ),
    )

    print("\n테스트 데이터 예측 시작...")
    # 테스트 데이터 예측
    test_data = pd.read_csv(TEST_DATA_PATH)
    test_data = preprocess_data(test_data, is_test=True)

    X_test = test_data[optimal_features]
    y_test_pred = predict_test_data(model, X_test, scaler)

    # 결과 저장
    save_predictions(y_test_pred, test_data["date_time"], SUBMISSION_PATH)
    print("\n작업 완료!")


if __name__ == "__main__":
    main()
