# 붓꽃 품종 예측
# 범주형 데이터 ->  분류 모델(classification)

import tensorflow as tf
import pandas as pd

file_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(file_path)

# one-hot encoding
# 범주형 데이터는 기존에 했던 회귀와는 다르게 결과로 도출해야 하는 것이 숫자값이 아님
# 공식에 숫자값이 아닌것을 넣을 수 없으므로 결과로 도출될 값들을 숫자값들로 변환 가능하도록 처리
iris = pd.get_dummies(iris)
print(iris.head())

# iris 독립, 종속 변수 분리
independent_var = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dependent_var = iris[['품종_setosa', '품종_versicolor', '품종_virginica']]

print(independent_var.shape, dependent_var.shape)

# 모델 구조 생성
X = tf.keras.layers.Input(shape=[4])
# softmax: 도출되는 결과값의 범위를 조절하는 활성화함수. 보통 분류 모델에서 사용됨
Y = tf.keras.layers.Dense(3, activation='softmax')(X)
model = tf.keras.models.Model(X, Y)
# 문제에 맞는 loss 유형 지정
# cross entropy: 보통 분류 모델에서 사용됨
# "metrics='accuracy'": 분류 모델에서는 loss 보다는 정확도가 사람이 보기에 더 편한 지표
model.compile(loss='categorical_crossentropy',
              metrics='accuracy')

# 모델 학습(fit)
model.fit(independent_var, dependent_var, epochs=1000)

# 모델 이용
print(model.predict(independent_var[-5:]))   # 모델의 예측값
print(dependent_var[-5:])                    # 실제값

# 모델 확인
print(model.get_weights())
