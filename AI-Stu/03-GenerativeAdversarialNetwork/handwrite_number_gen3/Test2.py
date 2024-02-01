import torch
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
])

# 加载真实数据集
true_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

totle_preview_size = 50
num_cols = 10  # 列
num_rows = int(totle_preview_size / num_cols)  # 行
# 创建子图并显示图片
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), subplot_kw={'aspect': 'auto'})  # 调整图像大小
preview_index = 0
for img_tensor, label in true_dataset:
    col = preview_index % num_cols
    row = int(preview_index / num_cols)
    print("image shape-->", img_tensor.shape, " || max-->", torch.max(img_tensor).item(), " || min-->",
          torch.min(img_tensor).item())
    npImage = img_tensor.permute(1, 2, 0).numpy()
    axes[row, col].imshow(npImage, cmap='gray')
    axes[row, col].set_title(label)
    preview_index += 1
    if preview_index >= totle_preview_size:
        break
plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()
