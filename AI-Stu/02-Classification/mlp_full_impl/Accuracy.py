# 计算预测正确的数量
# 例如y_hat=[[0.2,0.3,0.5],[0.1,0.9,0],[0.1,0.01,0.99]]
# y=[2,1,1]
# 经过 y_hat = y_hat.argmax(axis=1)
# y_hat=[2, 1, 2]
# 所以计算正确的有两个（前面两个），计算不正确的有一个（最后一个）
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())