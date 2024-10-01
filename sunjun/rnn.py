import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 데이터 경로 설정
base_weather_path = './weather data/'
base_power_path = './power data/'

# 사용할 지역 목록
regions = ['gyeonggi', 'jeju', 'jeonnam', 'seoul', 'ulsan']

# 시퀀스 생성 함수 정의 (RNN 훈련용)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :-1]  # 기후 데이터 (입력)
        y = data[i+seq_length, -1]     # 전력 거래량 (타겟)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# LSTM 모델 정의 함수
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # 출력층 (전력 거래량 예측)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 각 지역별로 모델 훈련 및 예측
def train_and_predict(region):
    # 날씨 및 전력 데이터 불러오기
    weather_data = pd.read_csv(f'{base_weather_path}{region}_weather.csv', encoding='cp949')
    power_data = pd.read_csv(f'{base_power_path}{region}_power.csv', encoding='cp949')

    # 열 이름 공백 제거
    power_data.columns = power_data.columns.str.strip()
    weather_data.columns = weather_data.columns.str.strip()

    # 필요한 열만 선택하여 데이터 병합 (기온, 습도, 전력 거래량 등)
    weather_data = weather_data[['일시', '기온(°C)', '습도(%)', '풍속(m/s)']].copy()
    power_data = power_data[['거래일자', '전력거래량(MWh)']].copy()

    # 날짜 형식 맞추기 및 병합
    weather_data['일시'] = pd.to_datetime(weather_data['일시'])
    power_data['거래일자'] = pd.to_datetime(power_data['거래일자'])
    merged_data = pd.merge(weather_data, power_data, left_on='일시', right_on='거래일자').drop(['거래일자'], axis=1)

    # 결측치 처리 (단순한 결측값은 앞의 데이터로 채움)
    merged_data.fillna(method='ffill', inplace=True)

    # 데이터 스케일링 (Min-Max Scaling)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(merged_data[['기온(°C)', '습도(%)', '풍속(m/s)', '전력거래량(MWh)']])

    # 시퀀스 생성
    SEQ_LENGTH = 24  # 하루 단위로 데이터를 사용할 경우 24시간 시계열
    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    # 데이터셋을 훈련/검증으로 나누기
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM 모델 빌드 및 훈련
    model = build_lstm_model(input_shape=(SEQ_LENGTH, X.shape[2]))
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # 예측 수행
    y_pred = model.predict(X_test)

    # 스케일 복원 (전력 거래량에 해당하는 열만 inverse_transform)
    # 스케일 복원 (전력 거래량에 해당하는 열만 inverse_transform)
    # 스케일링된 X_test에서 마지막 열(전력거래량)을 복원할 때 전력거래량만 추출하여 역변환 수행
    y_test_rescaled = scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_test), X_test.shape[2])), y_test.reshape(-1, 1)], axis=1))[:, -1]
    y_pred_rescaled = scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_pred), X_test.shape[2])), y_pred], axis=1))[:, -1]

    # 예측 결과와 실제 값 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_rescaled, label='Actual Power Usage (MWh)')
    plt.plot(y_pred_rescaled, label='Predicted Power Usage (MWh)')
    plt.title(f"Actual vs Predicted Power Usage in {region.capitalize()}")
    plt.xlabel('Time Steps')
    plt.ylabel('Power Usage (MWh)')
    plt.legend()
    plt.show()

    # 예측 결과 반환 (전국 데이터에 포함)
    return merged_data, y_pred_rescaled, y_test_rescaled

# 전체 전국 데이터를 담을 리스트
national_X = []
national_y_pred = []
national_y_test = []

# 각 지역별로 모델 훈련 및 예측 수행
for region in regions:
    merged_data, y_pred_rescaled, y_test_rescaled = train_and_predict(region)
    national_X.append(merged_data)
    national_y_pred.append(y_pred_rescaled)
    national_y_test.append(y_test_rescaled)

# 전국 데이터를 하나의 시계열로 결합하여 시각화
national_y_pred = np.concatenate(national_y_pred)
national_y_test = np.concatenate(national_y_test)

# 전국 예측 결과와 실제 값 시각화
plt.figure(figsize=(10, 6))
plt.plot(national_y_test, label='Actual Power Usage (MWh)')
plt.plot(national_y_pred, label='Predicted Power Usage (MWh)')
plt.title("Actual vs Predicted Power Usage in National")
plt.xlabel('Time Steps')
plt.ylabel('Power Usage (MWh)')
plt.legend()
plt.show()
