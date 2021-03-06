# 보스턴 집값 예측 with hidden layer

import tensorflow as tf
import pandas as pd

file_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(file_path)
print(boston.head())

# boston 독립, 종속 변수 분리
independent_var = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
dependent_var = boston[['medv']]    # 집값(중앙값 - 평균이 이상치일 때는 중앙값으로 집단의 대표값을 설정))

print(independent_var.shape, dependent_var.shape)   # (506, 13) (506, 1)

# 모델 구조 생성
X = tf.keras.layers.Input(shape=[13])
# swish: 최근 발표된 활성화 함수
H = tf.keras.layers.Dense(10, activation='swish')(X)
Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# 모델 확인
print(model.summary())

# 모델 학습(fit)
model.fit(independent_var, dependent_var, epochs=1000, verbose=0)
# model.fit(independent_var, dependent_var, epochs=10)

# 모델 이용
print(model.predict(independent_var[:5]))   # 모델의 예측값
print(dependent_var[:5])                    # 실제값

# 모델 확인
print(model.get_weights())
