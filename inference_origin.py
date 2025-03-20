import cv2
import numpy as np
import torch
from preprocessing.feature_extraction import extract_handcrafted_features
from models.lstm_model import FatigueLSTM
def predict(vis_path,nir_path,th_path,model_path):
    capvis = cv2.VideoCapture(vis_path)
    capnir = cv2.VideoCapture(nir_path)
    capth=cv2.VideoCapture(th_path)
    if not capvis.isOpened():
        print(f"Error: Cannot open video file {vis_path}")
        return
    if not capnir.isOpened():
        print(f"Error: Cannot open video file {nir_path}")
        return
    if not capth.isOpened():
        print(f"Error: Cannot open video file {th_path}")
        return
    
    #fps = int(capvis.get(cv2.CAP_PROP_FPS))
    while capvis.isOpened() and capnir.isOpened() and capth.isOpened():
        retvis, framevis = capvis.read()
        retnir, framenir = capnir.read()
        retth, frameth = capth.read()
        framevis_gray = cv2.cvtColor(cv2.resize(framevis, (640, 480)), cv2.COLOR_BGR2GRAY)
        framenir_gray = cv2.cvtColor(cv2.resize(framenir, (640, 480)), cv2.COLOR_BGR2GRAY)
        frameth_gray = cv2.cvtColor(cv2.resize(frameth, (640, 480)), cv2.COLOR_BGR2GRAY)

        # 将灰度图像堆叠成三通道图像
        combined_frame = cv2.merge([framevis_gray, framenir_gray, frameth_gray])

        # 提取特征
        handcrafted_feature_vector = extract_handcrafted_features(framevis)
        resnet_feature_vector = extract_resnet_features(combined_frame)
        combined_feature_vector= np.concatenate([handcrafted_feature_vector, resnet_feature_vector])
        model = FatigueLSTM(input_size=521, hidden_size=128, num_layers=2, num_classes=2)
        model.load_state_dict(torch.load(model_path))
        model.eval()



