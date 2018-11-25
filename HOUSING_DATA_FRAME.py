import pandas as pd
df = pd.read_csv('./housing.txt', sep='\s+')
df.columns = ['CRIM', 'ZN', 'indus','chas','nox','rm','age','dis','rad',
              'tax', 'ptatio','b','lstat','medv']
print(df.head())
import seaborn as sns
for columns in df.columns:
    sns.pairplot(df[df.columns],)
