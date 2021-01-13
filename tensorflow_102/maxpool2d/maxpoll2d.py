import tensorflow as tf
import pandas as pd

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()
print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)

# reshape
train_X = train_X.reshape(60000, 28, 28, 1)

# one hot encoding
one_hot_train_Y = pd.get_dummies(train_Y)

# CNN 모델 구조 생성
X = tf.keras.layers.Input(shape=[28, 28, 1])

H = tf.keras.layers.Conv2D(3, kernel_size=5, activation='swish')(X)  # 3 feature map
# Flatten으로 인해 param의 수가 너무 많아지지 않도록 하기 위해 pooling 사용
# Max Pooling은 각 filter마다 가장 큰 값을 뽑아내는 pooling
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(H)  # 6 feature map
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)
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
