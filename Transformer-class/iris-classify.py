import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import datasets        # 导入 scikit-learn 库中的 dataset 模块

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 全连接层，做映射，d_model维度映射到d_k*n_heads维度
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]


        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_heads * d_v)  # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn  # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):  # enc_inputs : [batch_size x source_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_inputs)
        return enc_outputs


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransformerModel, self).__init__()

        # 定义 Transformer 编码器，并指定输入维数和头数
        self.encoder = Encoder()
        # 定义全连接层，将 Transformer 编码器的输出映射到分类空间
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # 在序列的第2个维度（也就是时间步或帧）上添加一维以适应 Transformer 的输入格式
        x = x.unsqueeze(1)  # 120x1x4，相当于把每个样本的特征向量看成一个序列，序列长度为1，特征维度为4
        # 将输入数据流经 Transformer 编码器进行特征提取
        x = self.encoder(x)  # 120x1x4
        # 通过压缩第2个维度将编码器的输出恢复到原来的形状
        x = x.squeeze(1)  # 120x4
        # 将编码器的输出传入全连接层，获得最终的输出结果
        x = self.fc(x)  # 120x3
        return x




def iris_data():
    iris = datasets.load_iris()
    X = iris.data  # 对应输入变量或属性（features），含有4个属性：花萼长度、花萼宽度、花瓣长度 和 花瓣宽度
    y = iris.target  # 对应目标变量（target），也就是类别标签，总共有3种分类

    # 把数据集按照80:20的比例来划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 对训练集和测试集进行归一化处理，常用方法之一是StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 将训练集和测试集转换为PyTorch的张量对象并设置数据类型
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).long()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).long()
    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    d_model = 4  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 1  # number of Encoder of Decoder Layer
    n_heads = 1  # number of heads in Multi-Head Attention
    X_train, X_test, y_train, y_test = iris_data()

    model = TransformerModel(input_size=4, num_classes=3)

    print("定义损失函数和优化器")
    # 定义损失函数（交叉熵损失）和优化器（Adam）
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("训练模型")
    # 训练模型，对数据集进行多次迭代学习，更新模型的参数
    num_epochs = 500
    for epoch in range(num_epochs):
        # 前向传播计算输出结果
        outputs = model(X_train)

        loss = criterion(outputs, y_train)

        # 反向传播，更新梯度并优化模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印每10个epoch的loss值
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("测试模型")
    # 测试模型的准确率
    with torch.no_grad():
        # 对测试数据集进行预测，并与真实标签进行比较，获得预测
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Test Accuracy: {accuracy:.2f}')