import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class FatigueDataset(Dataset):
    def __init__(self, csv_file, sequence_length=30):
        """
        初始化数据集
        :param csv_file: CSV 文件路径
        :param sequence_length: 序列长度（LSTM 的时间步长）
        """
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length

        # 确保所有特征列都是数值类型
        self.features = self.data.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').values  # 特征列（去掉 frame_number 和 label）
        self.labels = self.data["label"].values  # 标签列

    def __len__(self):
        # 返回数据集的总长度
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        # 返回一个序列的特征和标签
        sequence_features = self.features[idx:idx + self.sequence_length]
        sequence_label = self.labels[idx + self.sequence_length - 1]  # 取序列最后一个时间步的标签

        # 确保数据是浮点类型
        sequence_features = sequence_features.astype(np.float32)
        sequence_label = np.array(sequence_label, dtype=np.int64)

        return torch.tensor(sequence_features, dtype=torch.float32), torch.tensor(sequence_label, dtype=torch.long)

# 示例用法
if __name__ == "__main__":
    dataset = FatigueDataset("output_features2.csv", sequence_length=30)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_features, batch_labels in dataloader:
        print(batch_features.shape)  # 形状: (batch_size, sequence_length, feature_dim)
        print(batch_labels.shape)    # 形状: (batch_size,)
        break