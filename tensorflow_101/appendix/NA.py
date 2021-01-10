import pandas as pd

file_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris2.csv'
iris = pd.read_csv(file_path)
print(iris.head())

# NaN이 없는지 파악
print(iris.isna().sum())
'''
꽃잎길이     0
꽃잎폭      1
꽃받침길이    0
꽃받침폭     0
품종       0
'''
print(iris.tail())

# NaN이 있는 경우 해당 데이터를 모델에 그대로 넣으면 에러가 발생함
# NaN값을 적절한 값으로 대체해서 넣어줘야 함
mean = iris['꽃잎폭'].mean()
iris['꽃잎폭'] = iris['꽃잎폭'].fillna(mean)
print(iris.tail())
