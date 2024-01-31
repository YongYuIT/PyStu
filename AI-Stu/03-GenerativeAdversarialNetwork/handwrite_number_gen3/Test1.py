# 计算感受野变化
import torch
import matplotlib.pylab as plt
from torch import nn

# 构造28*28的全一张量
srcImg = torch.ones(28, 28)
print('srcImg-->', srcImg)
plt.imshow(srcImg, cmap='gray')
plt.colorbar()
plt.show()
# 3*3卷积核，填充1，步幅1
conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
conv1.weight.data = torch.ones((1, 1, 3, 3))
conv1.bias.data = torch.zeros(1)
c1_srcImg = conv1.forward(srcImg.view(1, srcImg.size(1), srcImg.size(1))).detach()
c1_srcImg = c1_srcImg.view(c1_srcImg.size(1), c1_srcImg.size(1))
print('ketnel-->', conv1.weight.data.shape, "-->", conv1.weight.data)
print('bias-->', conv1.bias.data.shape, "-->", conv1.bias.data)
print('c1_srcImg-->', c1_srcImg)
plt.imshow(c1_srcImg, cmap='gray')
plt.colorbar()
plt.show()
# 为了更好观察感受野变化，这里把平均汇聚改成求和汇聚
pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
p1_srcImg = pool1.forward(c1_srcImg.view(1, c1_srcImg.size(1), c1_srcImg.size(1)))
p1_srcImg = p1_srcImg.view(p1_srcImg.size(1), p1_srcImg.size(1)) * 4
print('p1_srcImg-->', p1_srcImg)
plt.imshow(p1_srcImg, cmap='gray')
plt.colorbar()
plt.show()
# 5*5卷积核，填充1，步幅2
conv2 = nn.Conv2d(1, 1, kernel_size=5, padding=2)
conv2.weight.data = torch.ones((1, 1, 3, 3))
conv2.bias.data = torch.zeros(1)
c2_srcImg = conv2.forward(p1_srcImg.view(1, p1_srcImg.size(1), p1_srcImg.size(1))).detach()
c2_srcImg = c2_srcImg.view(c2_srcImg.size(1), c2_srcImg.size(1))
print('ketnel-->', conv2.weight.data.shape, "-->", conv2.weight.data)
print('bias-->', conv2.bias.data.shape, "-->", conv2.bias.data)
print('c2_srcImg-->', c2_srcImg)
plt.imshow(c2_srcImg, cmap='gray')
plt.colorbar()
plt.show()
# 为了更好观察感受野变化，这里把平均汇聚改成求和汇聚
pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
p2_srcImg = pool2.forward(c2_srcImg.view(1, c2_srcImg.size(1), c2_srcImg.size(1)))
p2_srcImg = p2_srcImg.view(p2_srcImg.size(1), p2_srcImg.size(1)) * 4
print('p2_srcImg-->', p2_srcImg)
plt.imshow(p2_srcImg, cmap='gray')
plt.colorbar()
plt.show()
