import Accumulator as thk_accumulator
import Accuracy as thk_accuracy
import Animator as thk_animator
import DataLoader as thk_dataLoader
import ImageShow as thk_imageShow


# 训练模型一个迭代周期（定义见第3章）
def train_epoch_ch3(model, train_iter):
    # 训练损失总和、训练准确度总和、样本数
    metric = thk_accumulator.Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = model.net(X)
        l = model.loss(y_hat, y)
        l.sum().backward()
        # 执行SGD
        model.updater()
        metric.add(float(l.sum()), thk_accuracy.accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


# 训练模型（定义见第3章）
def train_ch3(model, train_iter, test_iter):  # @save
    animator = thk_animator.Animator(xlabel='epoch', xlim=[1, model.num_epochs], ylim=[0.3, 0.9],
                                     legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(model.num_epochs):
        train_metrics = train_epoch_ch3(model, train_iter)
        test_acc = thk_accumulator.evaluate_accuracy(model.net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


# 预测标签（定义见第3章）
def predict_ch3(model, test_iter, n=6):  # @save
    for X, y in test_iter:
        break
    trues = thk_dataLoader.get_fashion_mnist_labels(y)
    preds = thk_dataLoader.get_fashion_mnist_labels(model.net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    thk_imageShow.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
