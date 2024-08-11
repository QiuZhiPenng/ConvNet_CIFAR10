import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

# 1.获取输入图片
image_path = "./imgs/airplane.png"
# 转为PIL image
image = Image.open(image_path)
# 转为RGB三通道
image = image.convert("RGB")

# print(image)

# 2.将size转为32*32，符合模型的输入
trans = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                       torchvision.transforms.ToTensor()])
image = trans(image)

# print(image.shape)

# torch.load时需要指定模型
class myModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 3.加载已经训练好的模型
my_model = torch.load("my_model_20_gpu2.pth")
# gpu映射为cpu运行
# my_model = torch.load("my_model_10.pth", map_location=torch.device("cpu"))
# print(my_model)

# 使用gpu需要转为cuda格式
image = image.to("cuda")
# 增加batch_size为1
image = torch.reshape(image, (1, 3, 32, 32))

# 模型设置为验证模式
my_model = my_model.eval()
# 设置为无梯度，节约内存，提高性能
with torch.no_grad():
    # 4.验证
    output = my_model(image)

print(f"预测的结果为：{output}")

class_index = output.argmax(1).item()
print(f"预测的分类id为：{class_index}")

# class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

class_to_idx = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print(f"分类id对应的分类为：{class_to_idx[class_index]}")
