import torch.nn as nn
import torch

class FatigueLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        初始化 LSTM 模型
        :param input_size: 输入特征维度
        :param hidden_size: 隐藏层维度
        :param num_layers: LSTM 层数
        :param num_classes: 输出类别数
        """
        super(FatigueLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # 全连接层

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))  # 输出形状: (batch_size, sequence_length, hidden_size)

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # 形状: (batch_size, hidden_size)

        # 全连接层
        out = self.fc(out)  # 形状: (batch_size, num_classes)
        return out


# 示例用法
if __name__ == "__main__":
    # 定义模型
    input_size = 521  # 输入特征维度
    hidden_size = 128  # 隐藏层维度
    num_layers = 2  # LSTM 层数
    num_classes = 2  # 输出类别数（0 或 1）
    model = FatigueLSTM(input_size, hidden_size, num_layers, num_classes)

    # 测试模型
    batch_size = 32
    sequence_length = 30
    test_input = torch.randn(batch_size, sequence_length, input_size)
    output = model(test_input)
    print(output)
    print(output.shape)  # 形状: (batch_size, num_classes)