import pandas as pd

file_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade = pd.read_csv(file_path)

file_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(file_path)

file_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(file_path)

# 데이터 모양 확인
print(lemonade.shape)
print(boston.shape)
print(iris.shape)

# 칼럼 이름 출력
print(lemonade.columns)
print(boston.columns)
print(iris.columns)

# lemonade 독립, 종속 변수 분리
independent_var = lemonade[['온도']]
dependent_var = lemonade[['판매량']]

print(independent_var.shape, dependent_var.shape)

# boston 독립, 종속 변수 분리
independent_var = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
dependent_var = boston[['medv']]

print(independent_var.shape, dependent_var.shape)

# iris 독립, 종속 변수 분리
independent_var = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dependent_var = iris[['품종']]

print(independent_var.shape, dependent_var.shape)

print(lemonade.head())
