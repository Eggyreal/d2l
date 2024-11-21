## pandas基本用法

### 主要用于处理dataframe的数据，实践中除了读取csv excel文件好像也没什么用了

和tensor差不多的创建方法，实践基本没有用

```python
import pandas as pd

data = {
  'age': [25, 32, 40],
    'salary': [50000, 62000, 70000],
    'name': ['Alice', 'Bob', 'Charlie'],
    'experience': [2, 5, 8]
}
data = pd.DataFrame(data)
```

主要用法，加载csv文件

```python
df = pd.read_csv('filename.csv')    #生成dataframe文件
df.iloc(line, column)  #train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]展示0~4行，左数0~3列，右数0~2列
```

生成索引		结果：RangeIndex(start=0, stop=3, step=1)

```python
#生成索引
data.index 
```

也是很重要的用法，可以用来清洗数据

```python
numeric_features = all_features.dtypes[all_features.dtypes != 'object']
#取出所有数据特征

df.apply(func, axis=0)    #对每一个数据执行func，axis=0（默认）针对某一列，例如：
df.apply(lamba x: (x-x.mean()) / x.std(), axis=0)    #其中的x.mean() x.std()都是针对某一列来讲的
```

