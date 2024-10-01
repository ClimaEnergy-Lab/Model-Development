import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# 데이터 경로 설정
base_power_path = './power data/'
regions = ['gyeonggi', 'jeju', 'jeonnam', 'seoul', 'ulsan']

# LSTM을 사용한 학습 및 예측 함수
def train_and_predict_lstm(region):
    # 전력 데이터 불러오기
    power_data = pd.read_csv(f'{base_power_path}{region}_power.csv', encoding='cp949')

    # 열 이름 공백 제거
    power_data.columns = power_data.columns.str.strip()

    # 필요한 열만 선택
    power_data = power_data[['거래일자', '전력거래량(MWh)']].copy()

    # 날짜 형식 변환
    power_data['거래일자'] = pd.to_datetime(power_data['거래일자'])
    power_data.set_index('거래일자', inplace=True)

    # 중복된 날짜 제거
    power_data = power_data[~power_data.index.duplicated(keep='first')]

    # 결측치 처리 (전력 사용량이 없는 경우는 0으로 채움)
    power_data.fillna(0, inplace=True)

    # 데이터 스케일링 (MinMaxScaler 사용)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(power_data[['전력거래량(MWh)']])

    # 학습 및 테스트 데이터 분할 (80% 학습, 20% 테스트)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # 시계열 데이터를 LSTM 입력에 맞는 형태로 변환 (시퀀스 길이 설정: 30일)
    def create_sequences(data, seq_length=30):
        sequences = []
        labels = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
            labels.append(data[i+seq_length])
        return np.array(sequences), np.array(labels)

    seq_length = 30
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # LSTM 모델 정의
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(seq_length, 1)))
    model.add(Dense(1))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 모델 학습
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # 예측 수행
    predictions = model.predict(X_test)

    # 예측값과 실제값을 스케일 복원
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test_rescaled)), y_test_rescaled, label='Actual Power Usage (MWh)', color='blue')
    plt.plot(range(len(predictions_rescaled)), predictions_rescaled, label='Predicted Power Usage (MWh)', color='red')
    plt.title(f"Actual vs Predicted Power Usage in {region.capitalize()} (LSTM)")
    plt.xlabel('Time')
    plt.ylabel('Power Usage (MWh)')
    plt.legend()
    plt.show()

    # 성능 평가 지표 출력
    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    print(f"{region.capitalize()} MSE: {mse}")

    return y_test_rescaled, predictions_rescaled

# 전국 데이터를 위한 결과 저장 리스트
national_actual = []
national_pred = []

# 각 지역별로 LSTM 모델을 적용하여 예측 수행
for region in regions:
    test_data, predictions = train_and_predict_lstm(region)
    national_actual.append(test_data)
    national_pred.append(predictions)

# 전국 데이터를 결합하여 시각화
national_actual = np.concatenate(national_actual)
national_pred = np.concatenate(national_pred)

plt.figure(figsize=(10, 6))
plt.plot(national_actual, label='Actual Power Usage (MWh)', color='blue')
plt.plot(national_pred, label='Predicted Power Usage (MWh)', color='red')
plt.title("Actual vs Predicted Power Usage in National (LSTM)")
plt.xlabel('Time')
plt.ylabel('Power Usage (MWh)')
plt.legend()
plt.show()

# 전국 성능 평가
national_mse = mean_squared_error(national_actual, national_pred)
print(f"National MSE: {national_mse}")
