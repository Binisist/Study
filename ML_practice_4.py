import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LeakyReLU

# 데이터 다운로드 및 읽기
url = "https://data.cdc.gov/api/views/9j2v-jamp/rows.csv?accessType=DOWNLOAD"
urllib.request.urlretrieve(url, filename="Death_rates_for_suicide__by_sex__race__Hispanic_origin__and_age__United_States.csv")
data = pd.read_csv('Death_rates_for_suicide__by_sex__race__Hispanic_origin__and_age__United_States.csv', encoding='latin1')

# 입력(X)과 출력(y) 데이터 설정
data_X = data[["STUB_NAME_NUM", "STUB_LABEL_NUM", "YEAR_NUM", "AGE_NUM"]].values
data_y = data['ESTIMATE'].values

# NaN 값을 평균값으로 대체
mean_value = np.nanmean(data_y)
data_y = np.where(np.isnan(data_y), mean_value, data_y)

# 데이터 정규화
scaler = MinMaxScaler()
data_X = scaler.fit_transform(data_X)

# 훈련 데이터와 테스트 데이터로 분할 (8:2)
(X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size=0.8, random_state=1)

# Sequential 모델 생성 및 레이어 추가
model = Sequential()
model.add(Dense(256, input_dim=4, kernel_regularizer=l2(0.01)))  # 은닉층 크기 변경, 규제 강도 변경
model.add(LeakyReLU(alpha=0.2))  # Leaky ReLU 적용
model.add(Dense(256, kernel_regularizer=l2(0.01)))  # 은닉층 크기 변경, 규제 강도 변경
model.add(LeakyReLU(alpha=0.2))  # Leaky ReLU 적용
model.add(Dense(256, kernel_regularizer=l2(0.01)))  # 은닉층 추가, 규제 강도 변경
model.add(LeakyReLU(alpha=0.2))  # Leaky ReLU 적용
model.add(Dense(256, kernel_regularizer=l2(0.01)))  # 은닉층 추가, 규제 강도 변경
model.add(LeakyReLU(alpha=0.2))  # Leaky ReLU 적용
model.add(Dense(1, activation='linear'))  # 출력 레이어는 활성화 함수를 사용하지 않음

# 옵티마이저 변경 (RMSprop 사용)
optimizer = RMSprop(learning_rate=0.001)

# 모델 컴파일
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error', 'mean_absolute_error'])  # MAE 평가지표 추가

# 모델 학습
history = model.fit(X_train, y_train, epochs=500, batch_size=1, validation_data=(X_test, y_test))

# 손실 값 시각화
epochs = range(1, len(history.history['mean_squared_error']) + 1)
plt.plot(epochs, history.history['mean_squared_error'])
plt.plot(epochs, history.history['val_mean_squared_error'])
plt.title('model loss')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

# 테스트 결과 평가 및 출력
evaluation = model.evaluate(X_test, y_test)
print("\n테스트 MSE: %.4f" % evaluation[1])
print("테스트 MAE: %.4f" % evaluation[2])
print("테스트 R^2 스코어: %.4f" % (1 - evaluation[1] / np.var(y_test)))
