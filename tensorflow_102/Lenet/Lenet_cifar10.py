# Lenet 5

import tensorflow as tf
import pandas as pd

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.cifar10.load_data()
print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)

# one hot encoding
# cifat10은 train_Y의 형태가 2차원이므로 reshape한 후에 one-hot-encoding
one_hot_train_Y = pd.get_dummies(train_Y.reshape(50000))

# Lenet 모델 구조 생성
X = tf.keras.layers.Input(shape=[32, 32, 3])

H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')
print(model.summary())

# 모델 학습
model.fit(train_X, one_hot_train_Y, epochs=10)

# 모델 이용
pred = model.predict(test_X[:5])
print(pd.DataFrame(pred).round(2))

print("================================")
print(test_Y[:5])
