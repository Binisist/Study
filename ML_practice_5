##ML_practice_5

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Data get
dataset_path = keras.utils.get_file("wine_quality_red.csv", "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")

column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol', 'quality']
raw_dataset = pd.read_csv(dataset_path, names=column_names, header=0, delimiter=';')
dataset = raw_dataset.copy()

# Data 정제
dataset = dataset.dropna()

# Data 분할
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 레이블 분리
train_labels = train_dataset.pop('quality')
test_labels = test_dataset.pop('quality')

# 데이터 정규화 및 전처리 개선
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# 모델 만들기
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.leaky_relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(128, activation=tf.nn.leaky_relu),
        layers.Dense(128, activation=tf.nn.leaky_relu),
        layers.Dense(128, activation=tf.nn.leaky_relu),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse', 'accuracy', 'msle', 'mape'])
    return model

model = build_model()
model.summary()

# 모델 훈련
history = model.fit(
    normed_train_data, train_labels,
    epochs=1000, validation_split=0.2, verbose=0
)

# 모델 성능 확인
loss, mae, mse, accuracy, msle, mape = model.evaluate(normed_test_data, test_labels, verbose=2)

print("테스트 세트의 평균 절대 오차: {:5.2f} quality".format(mae))
print("테스트 세트의 정확도: {:5.2f}%".format(accuracy * 100))
print("테스트 세트의 평균 제곱 로그 오차: {:5.2f}".format(msle))
print("테스트 세트의 평균 절대 백분율 오차: {:5.2f}%".format(mape * 100))

# 훈련 과정 시각화
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plt.figure(figsize=(8, 12))

plt.subplot(2, 1, 1)
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [Quality]')
plt.plot(hist['epoch'], hist['mae'], label='Train Error')
plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
plt.ylim([0, 5])
plt.legend()

plt.subplot(2, 1, 2)
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error [Quality^2]')
plt.plot(hist['epoch'], hist['mse'], label='Train Error')
plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
plt.ylim([0, 20])
plt.legend()
plt.show()

# 예측
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Quality]')
plt.ylabel('Predictions [Quality]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
