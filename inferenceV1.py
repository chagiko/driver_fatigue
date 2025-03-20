import cv2
import numpy as np
import torch
import dlib
from torchvision import models, transforms
from preprocessing.create_dataset import extract_handcrafted_features, extract_resnet_features
from models.lstm_model import FatigueLSTM


def load_emg_signal(frame_idx):
    """
    读取对应帧的肌电信号数据，假设肌电信号为 N×1000 的数组
    :param frame_idx: 当前帧索引
    :return: 形状为 (N, 1000) 的 NumPy 数组
    """
    # TODO: 这里需要替换成你实际的肌电信号读取逻辑
    return np.random.rand(8, 1000)  # 假设有 8 个通道，每通道 1000 个数据点

def predict(vis_path, nir_path, th_path, model_path):
    capvis = cv2.VideoCapture(vis_path)
    capnir = cv2.VideoCapture(nir_path)
    capth = cv2.VideoCapture(th_path)

    if not capvis.isOpened() or not capnir.isOpened() or not capth.isOpened():
        print("Error: Cannot open one or more video files.")
        return
    # 加载 ResNet
    resnet = models.resnet18(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层
    resnet.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载 LSTM 模型
    model = FatigueLSTM(input_size=521, hidden_size=128, num_layers=2, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 提取特征
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # 缓存最近 30 帧的特征
    feature_window = []
    frame_idx = 0  # 用于肌电信号索引

    while capvis.isOpened() and capnir.isOpened() and capth.isOpened():
        retvis, framevis = capvis.read()
        retnir, framenir = capnir.read()
        retth, frameth = capth.read()
        if not (retvis and retnir and retth):
            break  # 读取失败时退出循环

        # 预处理：转换为灰度图并堆叠
        framevis_gray = cv2.cvtColor(cv2.resize(framevis, (640, 480)), cv2.COLOR_BGR2GRAY)
        framenir_gray = cv2.cvtColor(cv2.resize(framenir, (640, 480)), cv2.COLOR_BGR2GRAY)
        frameth_gray = cv2.cvtColor(cv2.resize(frameth, (640, 480)), cv2.COLOR_BGR2GRAY)
        combined_frame = cv2.merge([framevis_gray, framenir_gray, frameth_gray])


        handcrafted_feature_vector = extract_handcrafted_features(framevis, face_detector, landmark_predictor)
        resnet_feature_vector = extract_resnet_features(combined_frame, resnet, device)

        # 拼接特征向量
        feature_vector = np.concatenate([handcrafted_feature_vector, resnet_feature_vector])

        # 维护时间窗口
        feature_window.append(feature_vector)
        if len(feature_window) > 30:
            feature_window.pop(0)  # 只保留最近 30 帧

        # 读取当前帧对应的肌电信号
        emg_signal = load_emg_signal(frame_idx)  # 形状 (N, 1000)
        frame_idx += 1  # 更新帧索引

        if len(feature_window) == 30:  # 只有当累积 30 帧后才开始预测
            X = torch.tensor(np.array(feature_window), dtype=torch.float32).unsqueeze(0)  # 形状: (1, 30, 521)
            with torch.no_grad():
                predictions = model(X)  # 形状: (1, num_classes)

            last_prediction = predictions  # 取最后一帧的预测结果
            print(last_prediction)
            #predicted_class = torch.argmax(last_prediction[0, -1]).item()  # 0=清醒，1=疲劳
            predicted_class = torch.argmax(last_prediction[0]).item()# 0=清醒，1=疲劳
            # 返回当前帧的数据
            yield framevis, framenir, frameth, emg_signal, predicted_class

    capvis.release()
    capnir.release()
    capth.release()

if __name__ == "__main__":
    vis_path = "visible.mp4"  # 视频文件目录   
    nir_path = "visible.mp4"
    th_path = "visible.mp4"
    model_path="checkpoints/best_model.pth"
    for framevis, framenir, frameth, emg_signal, predicted_class in predict(vis_path, nir_path, th_path, model_path):
        print(f"Predicted Class: {predicted_class}")
        print(f"EMG Signal Shape: {emg_signal.shape}")  # (N, 1000)
        
        # 显示图像
        #cv2.imshow("Visible Light", framevis)
        #cv2.imshow("Near Infrared", framenir)
        #cv2.imshow("Thermal Infrared", frameth)
        
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    #cv2.destroyAllWindows()
    #predict(vis_path,nir_path,th_path,model_path)
    

