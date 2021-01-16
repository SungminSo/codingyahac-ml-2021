import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# 로컬 데이터 읽어들여 준비
paths = glob.glob('../../notMNIST_small/*/*.png')
paths = np.random.permutation(paths)
train_X = np.array([plt.imread(paths[i]) for i in range(len(paths))])
# if macOS: train_Y = np.array([paths[i].split('/')[-2] for i in range(len(paths))])
train_Y = np.array([paths[i].split('\\')[-2] for i in range(len(paths))])

print(paths[0])

print(train_X.shape, train_Y.shape)  # (18724, 28, 28) (18724,)

# reshape
train_X = train_X.reshape(18724, 28, 28, 1)

# one hot encoding
one_hot_train_Y = pd.get_dummies(train_Y)

print(train_X.shape, one_hot_train_Y.shape)  # (18724, 28, 28) (18724,)

# Lenet 모델 구조 생성
X = tf.keras.layers.Input(shape=[28, 28, 1])

H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X)
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
pred = model.predict(train_X[0:5])
print(pd.DataFrame(pred).round(2))

# 정답 확인
print(train_Y[0:5])
