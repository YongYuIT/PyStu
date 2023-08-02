import DataLoader as thk_dataLoader

train_iter, test_iter = thk_dataLoader.load_data_fashion_mnist(32, resize=64)

# 确保子进程不会再次执行主模块的代码
if __name__ == '__main__':
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
