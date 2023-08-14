import Accumulator as thk_accumulator
import Accuracy as thk_accuracy
import Animator as thk_animator


# 训练模型一个迭代周期（定义见第3章）
def train_epoch_ch3(model, train_iter):
    # 将模型设置为训练模式
    model.net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = thk_accumulator.Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = model.net(X)
        l = model.loss(y_hat, y)
        # 使用PyTorch内置的优化器和损失函数
        model.updater.zero_grad()
        l.mean().backward()
        model.updater.step()
        metric.add(float(l.sum()), thk_accuracy.accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


# 训练模型（定义见第3章）
def train_ch3(model, train_iter, test_iter):  # @save
    animator = thk_animator.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                                     xlim=[1, model.num_epochs], ylim=[1e-3, 1e2],
                                     legend=['train', 'test'])
    for epoch in range(model.num_epochs):
        train_epoch_ch3(model, train_iter)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (thk_accumulator.evaluate_loss(model.net, train_iter, model.loss),
                                     thk_accumulator.evaluate_loss(model.net, test_iter, model.loss)))
    print('weight:', model.net[0].weight.data.numpy())

