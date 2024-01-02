def countdown(n):
    while n > 0:
        yield n
        n -= 1

# 创建生成器对象
counter = countdown(5)

# 通过迭代获取生成的值
for num in counter:
    print(num)