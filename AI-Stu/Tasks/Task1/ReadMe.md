# 整体目标

使用全连接判断图片里面有没有狗

# 数据准备（DataPrepare）

1. Spider-Stu\AI_Task1_DataPrepare项目分次下载几百张图片
    * 第一次下载约200张狗狗的图片存在dog文件夹下
    * 第二次下载约200张猪的图片存在pig文件夹下
    * 第三次下载约200张小鸟的图片存在bird文件夹下
    * 第四次下载约200张美女的图片存在girl文件夹下
    * 第五此下载约200张蛇的图片存在snake文件夹下
2. 手动剔除无关图片，然后为每张图片生成唯一id，并作为文件名重命名图片
3. 生成一个cvs表格存储图片文件名（即id）和其对应的动物类型名称
4. 将这些图片和cvs整理到一个文件夹中得到原始数据

# 数据输入

参考：https://zh.d2l.ai/chapter_deep-learning-computation/read-write.html#id2

