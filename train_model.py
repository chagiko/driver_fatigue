import torch.optim as optim
from sklearn.metrics import accuracy_score
import torch
from dataset import FatigueDataset
from models.lstm_model import FatigueLSTM
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device="cpu", save_dir="checkpoints"):
    """
    训练函数
    :param model: LSTM 模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param num_epochs: 训练轮数
    :param device: 训练设备（"cpu" 或 "cuda"）
    :param save_dir: 模型权重保存目录
    """
    model.to(device)
    best_val_acc = 0.0  # 记录最佳验证集准确率
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录

    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        all_labels = []
        all_preds = []

        # 训练阶段
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失和准确率
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 计算训练集的平均损失和准确率
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # 保存最佳模型权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"Best model saved with Val Accuracy: {best_val_acc:.4f}")

def validate(model, val_loader, criterion, device):
    """
    验证函数
    :param model: LSTM 模型
    :param val_loader: 验证数据加载器
    :param criterion: 损失函数
    :param device: 训练设备（"cpu" 或 "cuda"）
    :return: 验证集的平均损失和准确率
    """
    model.eval()  # 设置为评估模式
    val_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 记录损失和准确率
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算验证集的平均损失和准确率
    val_loss /= len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    return val_loss, val_acc

# 示例用法
if __name__ == "__main__":
    # 定义数据集和数据加载器
    train_dataset = FatigueDataset("train_dataset.csv", sequence_length=30)
    val_dataset = FatigueDataset("test_dataset.csv", sequence_length=30)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    # 定义模型、损失函数和优化器
    model = FatigueLSTM(input_size=521, hidden_size=128, num_layers=2, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device=device, save_dir="checkpoints")