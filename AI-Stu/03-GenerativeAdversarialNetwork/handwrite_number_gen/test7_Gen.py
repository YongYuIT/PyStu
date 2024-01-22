from ModelDesign import FCGenModelDef as FCG_MD
import torch
import matplotlib.pyplot as plt
from ModelDesign import LeNetGPUModelDef as LN_MD

# 加载判别器模型
JustifyModel = LN_MD.LeNetGPUModelDef()
JustifyModel.loadModel("LeNetGPUModelDef")

# 加载生成器模型
GenModel = FCG_MD.FCGenModelDef(JustifyModel)
GenModel.loadModel("FCGenModelDef")

# 设置子图的行列数
simple_size = 30
num_cols = 10  # 列
num_rows = int(simple_size / num_cols)  # 行
# 创建子图并显示图片
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), subplot_kw={'aspect': 'auto'})  # 调整图像大小

JustifyModel.eval()
GenModel.eval()
with torch.no_grad():
    inputX = torch.normal(0, 1, (simple_size, 100)).to('cuda')
    genImgs = GenModel(inputX)
    loss = GenModel.lossTensor(genImgs)
    genImgs = genImgs.to('cpu')
    loss = loss.to('cpu')
    for index in range(genImgs.size(0)):
        col = index % num_cols
        row = int(index / num_cols)
        genImg = genImgs[index]
        npImage = genImg.permute(1, 2, 0).numpy()
        axes[row, col].imshow(npImage, cmap='gray')
        axes[row, col].set_title(round(loss[index].item(), 4))
plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()
