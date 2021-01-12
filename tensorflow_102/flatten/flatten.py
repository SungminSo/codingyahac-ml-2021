import tensorflow as tf
import pandas as pd

# 데이터 준비
(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()
print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)

# one-hot encoding
one_hot_train_Y = pd.get_dummies(train_Y)

print(train_X.shape, one_hot_train_Y.shape)

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
model.fit(train_X, one_hot_train_Y, epochs=10)

# 모델 이용
pred = model.predict(test_X[:5])
print(pd.DataFrame(pred).round(2))

print("================================")
print(test_Y[:5])
