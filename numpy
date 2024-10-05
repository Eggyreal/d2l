import numpy as np

#创建矩阵
array = np.array([1,2,3])
linspace_array = np.linspace(0, 1, 5)
random_array = np.random.rand(2, 3)  # 生成2行3列的随机数

#矩阵转置 ()^T
transposed_array = np.transpose(array)
#合并数组 
array1 = np.ones(2)  #(1,1)
array2 = np.zeros(3)  #(0,0,0)
concatenated_array = np.concatenate((array1, array2))    #(1,1,0,0,0)

#求和
total = np.sum(array)
# 计算均值
mean_value = np.mean(arr)
# 最小值\最大值
min_value = np.min(arr)  
max_value = np.max(arr) 
# 计算每个元素的平方根
sqrt_array = np.sqrt(arr)  
#矩阵元素相乘
array1 * array2
np.multiply(array1, array2)
#矩阵相乘
np.dot()  #只能二维用
np.matmul()
np.mm()
array1 @ array2
# 计算矩阵的逆
inverse_matrix = np.linalg.inv(matrix)  

any_true = np.any(arr > 1)  # 检查是否存在大于1的元素
all_true = np.all(arr > 0)  # 检查所有元素是否大于0



