import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# from model import *

# 定义训练的设备
device = torch.device("cuda")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练的设备:{device}")

# 1.准备数据集
train_dataset = torchvision.datasets.CIFAR10("../dataset2", train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10("../dataset2", train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)

# 数据集长度
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print(f"训练集的长度为:{train_dataset_size}")
print(f"测试集的长度为:{test_dataset_size}")

# 2.加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)


# 3.搭建神经网络模型
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
            # ReLU(),
            # 输出层：对于多分类问题，如果你使用的是nn.CrossEntropyLoss作为损失函数，通常不需要在最后一个全连接层后面添加ReLU激活函数，
            # 因为nn.CrossEntropyLoss内部已经包含了一个log_softmax操作，它首先将输入通过softmax转换为概率分布。
            # 如果你的输出层是用于回归任务，通常也不会使用ReLU，因为它会将负值置为零，这可能会丢失一些信息。
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

my_model = myModel()
# 使用gpu
# if torch.cuda.is_available():
#     my_model = my_model.cuda()
my_model = my_model.to(device)

# 4.创建损失函数
loss_fn = nn.CrossEntropyLoss()
# 使用gpu
# if torch.cuda.is_available():
#  loss_fn = loss_fn.cuda()
loss_fn = loss_fn.to(device)

# 5.创建优化器
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(my_model.parameters(), learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录训练的轮数
# epoch = 10
epoch = 20

# 添加tensorboard可视化
writer = SummaryWriter("../logs_train")

# 6.训练网络模型
# start_time = time.time()
for i in range(epoch):
    print(f"------第{i+1}轮训练开始------")

    # 训练步骤开始
     # 建议加上，This has any effect only on certain modules.
    my_model.train()
    for data in train_dataloader:
        # 1.获取训练数据
        imgs, targets = data
        # 使用gpu
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        imgs = imgs.to(device)
        targets = targets.to(device)
        # 2.训练数据放入模型
        outputs = my_model(imgs)
        # 3.计算损失
        loss = loss_fn(outputs, targets)
        # 梯度清零
        optimizer.zero_grad()
        # 4.反向传播，获得梯度
        loss.backward()
        # 5.优化，更新参数
        optimizer.step()

        # 训练次数加一（train_dataset的50000张图片/64为batch_size=781.25次=782次），每轮会训练782次
        total_train_step = total_train_step + 1
        # 训练次数每100次打印对应损失Loss
        if total_train_step % 100 == 0:
            # end_time = time.time()
            # print(end_time - start_time)
            print(f"训练次数:{total_train_step},对应损失Loss:{loss.item()}")
            # 可视化 训练次数对应损失Loss
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
     # 建议加上，This has any effect only on certain modules.
    my_model.eval()
     # 无梯度
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data_test in test_dataloader:
            imgs_test, targets_test = data_test

            # 将输入数据移动到GPU
            imgs_test = imgs_test.to('cuda')

            outputs_test = my_model(imgs_test)
            # 使用gpu
            # if torch.cuda.is_available():
            #     imgs_test = imgs_test.cuda()
            #     targets_test = targets_test.cuda()
            imgs_test = imgs_test.to(device)
            targets_test = targets_test.to(device)

            loss = loss_fn(outputs_test, targets_test)
            # 计算测试集的整体损失
            total_test_loss = total_test_loss + loss.item()
            # 计算测试集的整体正确率
            accuracy = (outputs_test.argmax(1) == targets_test).sum()
            total_test_accuracy = total_test_accuracy + accuracy

    print(f"每一轮结束后测试集的整体损失Loss:{total_test_loss}")
    print(f"每一轮结束后测试集上整体的正确率:{total_test_accuracy/test_dataset_size}")
    # 可视化 每一轮结束后测试集的整体损失Loss
    writer.add_scalar("test_loss_epoch", total_test_loss, i+1)
    # 可视化 每一轮结束后 测试集上整体的正确率
    writer.add_scalar("test_accuracy_epoch", total_test_accuracy/test_dataset_size, i+1)

    # 保存每一轮模型训练后的结果
    torch.save(my_model, f"my_model_{i+1}_gpu2.pth")
    print(f"第{i+1}轮模型已保存")

writer.close()
