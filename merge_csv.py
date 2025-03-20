import pandas as pd
import os
def merge_csv_files(csv_dir, output_csv_path):
    """
    合并 CSV 文件
    :param csv_dir: CSV 文件目录
    :param output_csv_path: 输出合并后的 CSV 文件路径
    """
    all_data = []

    for csv_name in os.listdir(csv_dir):
        if csv_name.endswith(".csv"):
            csv_path = os.path.join(csv_dir, csv_name)
            df = pd.read_csv(csv_path)
            all_data.append(df)

    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Merged dataset saved to {output_csv_path}")

# 示例用法
if __name__ == "__main__":
    csv_dir = "data/csv/XQY"  # CSV 文件目录
    output_csv_path = "merged_dataset.csv"  # 输出合并后的 CSV 文件路径
    merge_csv_files(csv_dir, output_csv_path)