import pandas as pd

def split_dataset_stratified(csv_file, train_ratio=0.8):
    """
    按分层抽样划分数据集
    :param csv_file: CSV 文件路径
    :param train_ratio: 训练集比例（默认 0.8）
    :return: 训练集和测试集的 DataFrame
    """
    data = pd.read_csv(csv_file)
    
    # 按标签分为非疲劳和疲劳两部分
    non_fatigue_data = data[data["label"] == 0]  # 非疲劳数据
    fatigue_data = data[data["label"] == 1]      # 疲劳数据
    
    # 对每一部分按比例划分训练集和测试集
    def split_by_time(df, train_ratio):
        split_index = int(len(df) * train_ratio)
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]
        return train_df, test_df
    
    # 划分非疲劳数据
    train_non_fatigue, test_non_fatigue = split_by_time(non_fatigue_data, train_ratio)
    
    # 划分疲劳数据
    train_fatigue, test_fatigue = split_by_time(fatigue_data, train_ratio)
    
    # 合并训练集和测试集
    train_data = pd.concat([train_non_fatigue, train_fatigue])
    test_data = pd.concat([test_non_fatigue, test_fatigue])
    
    # 打乱训练集和测试集（可选）
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    
    return train_data, test_data

# 示例用法
if __name__ == "__main__":
    csv_file = "merged_dataset.csv"
    train_data, test_data = split_dataset_stratified(csv_file, train_ratio=0.8)
    
    # 保存训练集和测试集
    train_data.to_csv("train_dataset.csv", index=False)
    test_data.to_csv("test_dataset.csv", index=False)
    print("Train and test datasets saved.")