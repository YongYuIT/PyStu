import numpy as np

max_degree = 20
print(np.arange(max_degree))
print(np.arange(max_degree).reshape(1, -1))

print("--------------------------------------------------")

# power：对数组中的元素进行指数运算

# 情况一：底数和指数一样多
testArray = np.array([[2, 3, 1], [-2, -3, -1]])
testPow = np.array([[1, 2, 3]])
print("case1-->", np.power(testArray, testPow))
print("--------------------------------------------------")

# 情况二：底数只有一个，指数多个
testArray = np.array([[2], [-2]])
testPow = np.array([[1, 2, 3]])
print("case2-->", np.power(testArray, testPow))
print("--------------------------------------------------")

# 情况三：底数多个，指数多个；其实是情况二的变形
testArray = np.array([2, 1, -1, 3, -3, 1]).reshape(6, 1)
print("testArray-->", testArray)
testPow = np.array([1, 2, 3, 4]).reshape(1, -1)
print("testPow-->", testPow)
print("case3-->", np.power(testArray, testPow))
