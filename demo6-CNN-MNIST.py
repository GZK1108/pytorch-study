import torch
import torch.nn as nn
import scipy.io as io
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1 , 10, 5)  # 10@24*24
        self.pool1 = nn.MaxPool2d(2, 2)  # 10@12*12
        self.conv2 = nn.Conv2d(10, 20, 3)  # 20@10*10
        self.pool2 = nn.MaxPool2d(2, 2)  # 20@5*5
        self.fc1 = nn.Linear(20*5*5, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 10@24*24
        out = self.pool1(out)  # 10@12*12
        out = self.conv2(out)  # 20@10*10
        out = self.pool2(out)  # 20@5*5
        out = out.view(in_size, -1)  # 320
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


def main():
    data = io.loadmat('MNISTData.mat')
    # 读取训练与测试数据
    D_Train = data['D_Train']  # 标识
    D_Test = data['D_Test']
    X_Train = data['X_Train']  # 数据
    X_Test = data['X_Test']
    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    epochs = 1
    for epoch in range(epochs):
        for i in range(X_Train.shape[2]):
            x = torch.from_numpy(X_Train[:, :, i]).float().unsqueeze(0).unsqueeze(0)
            d = torch.from_numpy(D_Train[:, i].reshape(1,-1)).float()
            optimizer.zero_grad()
            output = net(x)
            print(output, d)
            loss = nn.functional.cross_entropy(output, d)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'epoch:{epoch}, i:{i}')
    print('训练完成，测试数据\n')
    # 测试数据
    true = 0  # 用于统计正确分类的个数
    for i in range(X_Test.shape[2]):  # 第i个训练样本
        x = torch.from_numpy(X_Test[:, :, i]).float().unsqueeze(0).unsqueeze(0)
        d = torch.from_numpy(D_Test[:, i].reshape(1, -1)).float()
        output = net(x)
        maxindex = np.argmax(output.detach().numpy())
        if d[0][maxindex] != 0:
            true = true + 1
    print('正确率为：', true / X_Test.shape[2])  # 效果拉垮

main()