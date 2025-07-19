# import numpy as np
#
# # 允许加载包含对象的 .npy 文件
# data = np.load('../dataset3/train/signal/4ASK/4ASK_-2dB_3117.npy', allow_pickle=True)
#
# print(data)
#
#
# # 查看数据类型和形状
# print("数据类型:", data.dtype)
# print("数据形状:", data.shape)

import numpy as np

# 加载 .npy 文件
data = np.load('../dataset/train/signal/4ASK/4ASK_-2dB_3117.npy', allow_pickle=True)

# 查看内容
print("数据内容:")
print(data)

# 查看数据类型
print("\n数据类型:", data.dtype)

# 查看数据形状
print("数据形状:", data.shape)

# 查看数组元素类别（如果是对象数组）
# if data.dtype == np.object:
print("数组元素类别:", type(data))

# 检查数组维度
print("数组维度:", data.ndim)

# 检查内存字节数
print("占用内存字节数:", data.nbytes)
