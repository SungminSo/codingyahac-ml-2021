import tensorflow as tf
import pandas as pd

# 데이터 준비
(independent_var, dependent_var), _ = tf.keras.datasets.mnist.load_data()

# one-hot encoding
dependent_var = pd.get_dummies(dependent_var)

print(independent_var.shape, dependent_var.shape)

# 모델 구조 생성
X = tf.keras.layers.Input(shape=[28, 28])
# 60000x28x28 -> 60000x784
# independent_var = independent_var.reshape(60000, 784) 와 같은 효과
H = tf.keras.layers.Flatten()(X)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 모델 학습
model.fit(independent_var, dependent_var, epochs=10)

# 모델 이용
print("Predicions: ", model.predict(independent_var[:5]))