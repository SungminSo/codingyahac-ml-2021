import pandas as pd

file_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris2.csv'
iris = pd.read_csv(file_path)
print(iris.head())

# one-hot encoding
encoding_iris = pd.get_dummies(iris)
print(encoding_iris.head())
# pandas는 셀의 데이터가 숫자이면 범주형 데이터일지라도 숫자로 인식해서 one-hot 인크딩이 안됨
# 그럴 경우에는 해당 칼럼의 데이터를 직접적으로 변경해줘야 함

print(iris.dtypes)

# '품종' 타입을 범주형으로 변경
iris['품종'] = iris['품종'].astype('category')
print(iris.dtypes)

# one-hot encoding
encoding_iris = pd.get_dummies(iris)
print(encoding_iris.head())
