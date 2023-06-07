# 一些基本工具的试用，跟本项目没有直接关系

from tqdm import tqdm
import time

# 1. tqdm试用
print("try tqdm--------------------------------------------------------start")
array = range(5)
print(array)
# 创建一个循环，使用 tqdm 显示进度条
for i in tqdm(range(100)):
    time.sleep(0.1)  # 模拟耗时操作
print("try tqdm--------------------------------------------------------end")
