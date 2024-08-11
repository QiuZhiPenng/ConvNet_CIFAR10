# 3.搭建神经网络模型（习惯单独用一个文件）
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU


class myModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # 输入 -> [卷积层] -> [ReLU激活函数] -> [池化层] -> ...
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, 1, 2),
            # 卷积层提取特征后，通过ReLU激活函数引入非线性，这有助于网络学习更复杂的特征表示。
            ReLU(),
            # 池化层（尤其是最大池化）本身具有降低特征维度和提取主要特征的作用，但不会引入非线性。在池化层之后使用ReLU并不会增加额外的非线性特性。
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            ReLU(),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            # 隐藏层：如果你的网络中全连接层是隐藏层，即后面还有更多的全连接层或卷积层，通常在全连接层后面添加ReLU激活函数是有益的。
            # 这有助于引入非线性，使网络能够学习更复杂的特征表示。
            ReLU(),
            # 输出层：对于多分类问题，如果你使用的是nn.CrossEntropyLoss作为损失函数，通常不需要在最后一个全连接层后面添加ReLU激活函数，
            # 因为nn.CrossEntropyLoss内部已经包含了一个log_softmax操作，它首先将输入通过softmax转换为概率分布。
            # 如果你的输出层是用于回归任务，通常也不会使用ReLU，因为它会将负值置为零，这可能会丢失一些信息。
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 简单验证一下 10分类
# 测试输出形状是否符合要求
if __name__ == '__main__':
    my_model = myModel()
    input = torch.ones((64, 3, 32, 32), dtype=torch.float32)
    output = my_model(input)
    print(output.shape)
    # torch.Size([64, 10])