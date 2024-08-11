import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
# import time

# 1.准备数据集
train_dataset = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)

# 数据集长度
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print(f"训练数据集的长度为:{train_dataset_size}")
print(f"测试数据集的长度为:{test_dataset_size}")

# 2.加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)


# 3.搭建神经网络模型
my_model = myModel()

# 4.创建损失函数
# 使用交叉熵损失
loss_fn = nn.CrossEntropyLoss()

# 5.创建优化器
# learning_rate = 0.01
learning_rate = 1e-2
# 使用SGD
optimizer = torch.optim.SGD(my_model.parameters(), learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录训练的轮数
epoch = 10

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
        for data in test_dataloader:
            imgs_test, targets_test = data
            outputs_test = my_model(imgs_test)
            loss = loss_fn(outputs_test, targets_test)
            # 计算测试集的整体损失
            total_test_loss = total_test_loss + loss.item()
            # 计算测试集的整体正确率
            accuracy = (outputs_test.argmax(1) == targets_test).sum()
            total_test_accuracy = total_test_accuracy + accuracy

    print(f"每一轮结束后验证集的整体损失Loss:{total_test_loss}")
    print(f"每一轮结束后验证集上整体的正确率:{total_test_accuracy/test_dataset_size}")
    # 可视化 每一轮结束后验证集的整体损失Loss
    writer.add_scalar("test_loss_epoch", total_test_loss, i+1)
    # 可视化 每一轮结束后 验证集上整体的正确率
    writer.add_scalar("test_accuracy_epoch", total_test_accuracy/test_dataset_size, i+1)

    # 保存每一轮模型训练后的结果
    torch.save(my_model, f"my_model_{i+1}_cpu.pth")
    print(f"第{i+1}轮模型已保存")

writer.close()
