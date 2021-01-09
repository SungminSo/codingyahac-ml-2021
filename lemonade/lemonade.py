# 레몬에이드 판매 예측

import tensorflow as tf
import pandas as pd

# 데이터 준비
file_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade = pd.read_csv(file_path)

independent_var = lemonade[['온도']]
dependent_var = lemonade[['판매량']]

print(independent_var.shape, dependent_var.shape)

# 모델 구조 생성
X = tf.keras.layers.Input(shape=[1])    # shape[1]에서 "1"은 입력되는 독립변수의 개수
Y = tf.keras.layers.Dense(1)(X)         # Dense(1)에서 "1"은 출력되는 종속변수의 개수

model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# 모델 학습(fit)
model.fit(independent_var, dependent_var, epochs=10000, verbose=0)      # epochs: 학습 횟수, verbose: 콘솔 출력 여부

# 모델 이용
print("Predictions: ", model.predict([[15]]))
