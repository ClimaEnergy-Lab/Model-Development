import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 데이터 경로 설정
base_power_path = './power data/'
regions = ['gyeonggi', 'jeju', 'jeonnam', 'seoul', 'ulsan']


# 각 지역별 ARIMA 모델 적용 함수
def train_and_predict_arima(region):
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

    # 날짜 인덱스의 주기를 설정합니다.
    power_data = power_data.asfreq('D')

    # 결측치 처리 (전력 사용량이 없는 경우는 0으로 채움)
    power_data.fillna(0, inplace=True)

    # 데이터 스케일링 (표준화)
    scaler = StandardScaler()
    power_data['전력거래량(MWh)_scaled'] = scaler.fit_transform(power_data[['전력거래량(MWh)']])

    # ARIMA 모델 학습 및 예측
    train_size = int(len(power_data) * 0.8)
    train_data = power_data[:train_size]['전력거래량(MWh)_scaled']
    test_data = power_data[train_size:]['전력거래량(MWh)_scaled']

    model = ARIMA(train_data, order=(5, 1, 0))
    model_fit = model.fit()

    # 예측 수행
    predictions = model_fit.forecast(steps=len(test_data))

    # NumPy 배열로 변환 후 스케일 복원
    predictions = np.array(predictions)
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    test_data_rescaled = scaler.inverse_transform(test_data.values.reshape(-1, 1)).flatten()

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(power_data[train_size:].index, test_data_rescaled, label='Actual Power Usage (MWh)', color='blue')
    plt.plot(power_data[train_size:].index, predictions_rescaled, label='Predicted Power Usage (MWh)', color='red')
    plt.title(f"Actual vs Predicted Power Usage in {region.capitalize()}")
    plt.xlabel('Date')
    plt.ylabel('Power Usage (MWh)')
    plt.legend()
    plt.show()

    # 성능 평가 지표 출력
    mse = mean_squared_error(test_data_rescaled, predictions_rescaled)
    print(f"{region.capitalize()} MSE: {mse}")

    return test_data_rescaled, predictions_rescaled


# 전국 데이터를 위한 결과 저장 리스트
national_actual = []
national_pred = []

# 각 지역별로 ARIMA 모델을 적용하여 예측 수행
for region in regions:
    test_data, predictions = train_and_predict_arima(region)
    national_actual.append(test_data)
    national_pred.append(predictions)

# 전국 데이터를 결합하여 시각화
national_actual = np.concatenate(national_actual)
national_pred = np.concatenate(national_pred)

plt.figure(figsize=(10, 6))
plt.plot(national_actual, label='Actual Power Usage (MWh)', color='blue')
plt.plot(national_pred, label='Predicted Power Usage (MWh)', color='red')
plt.title("Actual vs Predicted Power Usage in National")
plt.xlabel('Date')
plt.ylabel('Power Usage (MWh)')
plt.legend()
plt.show()

# 전국 성능 평가
national_mse = mean_squared_error(national_actual, national_pred)
print(f"National MSE: {national_mse}")
