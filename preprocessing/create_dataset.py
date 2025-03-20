import cv2
import dlib
import numpy as np
from deepface import DeepFace
from imutils import face_utils
import torch
from torchvision import models, transforms
import csv
import random
import os
# 加载dlib模型
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 加载ResNet18模型
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]) 
resnet.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 眼睛长宽比
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# 嘴巴长宽比
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[0] - mouth[6])
    B = np.linalg.norm(mouth[3] - mouth[9])
    return B / A

# 头部姿态估计
def estimate_head_pose(shape):
    model_points = np.array([
        (0.0, 0.0, 0.0),  # 鼻尖
        (0.0, -330.0, -65.0),  # 下巴
        (-225.0, 170.0, -135.0),  # 左眼角
        (225.0, 170.0, -135.0),  # 右眼角
        (-150.0, -150.0, -125.0),  # 左嘴角
        (150.0, -150.0, -125.0),  # 右嘴角
    ], dtype="double")

    image_points = np.array([
        shape[30],  # 鼻尖
        shape[8],   # 下巴
        shape[36],  # 左眼角
        shape[45],  # 右眼角
        shape[48],  # 左嘴角
        shape[54],  # 右嘴角
    ], dtype="double")

    camera_matrix = np.array([
        [1000, 0, 320],  # fx, 0, cx
        [0, 1000, 240],  # 0, fy, cy
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # 假设无畸变

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    if not success:
        return np.zeros(3)  # 无法估计时返回默认值

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = euler_angles.flatten()
    return np.array([pitch, yaw, roll])

# 提取特征
# def extract_features(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detector(gray)
    
#     # 初始化默认特征向量
#     default_features = {
#         "eye_state": np.zeros(2),  # 眼睛状态 (睁眼/闭眼)
#         "mouth_state": np.zeros(2),  # 嘴巴状态 (张开/闭合)
#         "head_pose": np.zeros(3),  # 头部姿态 (pitch, yaw, roll)
#         "emotion": np.zeros(2),  # 情绪 (积极/消极)
#         "resnet_features": np.zeros(512)  # ResNet18 特征 (512维)
#     }
    
#     if len(faces) == 0:
#         # 如果没有检测到人脸，返回默认特征
#         print("No face detected. Returning default features.")
#         feature_vector = np.concatenate([
#             default_features["eye_state"],
#             default_features["mouth_state"],
#             default_features["head_pose"],
#             default_features["emotion"],
#             default_features["resnet_features"]
#         ])
#         print(f"Default feature vector shape: {feature_vector.shape}")
#         return feature_vector
    
#     for face in faces:
#         shape = landmark_predictor(gray, face)
#         shape = face_utils.shape_to_np(shape)
        
#         # 眼睛状态
#         left_eye = shape[36:42]
#         right_eye = shape[42:48]
#         eye_ratio = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
#         eye_state = np.array([1, 0]) if eye_ratio < 0.2 else np.array([0, 1])
#         #print(f"Eye state shape: {eye_state.shape}")  # 调试信息
        
#         # 嘴巴状态
#         mouth = shape[48:60]
#         mouth_ratio = mouth_aspect_ratio(mouth)
#         mouth_state = np.array([1, 0]) if mouth_ratio > 0.5 else np.array([0, 1])
#         #print(f"Mouth state shape: {mouth_state.shape}")  # 调试信息
        
#         # 头部姿态
#         head_pose = estimate_head_pose(shape)
#         #print(f"Head pose shape: {head_pose.shape}")  # 调试信息
        
#         # 表情识别
#         analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
#         emotion = analysis[0]["dominant_emotion"]
#         emotion_state = np.array([1, 0]) if emotion in ["happy", "neutral"] else np.array([0, 1])
#         #print(f"Emotion state shape: {emotion_state.shape}")  # 调试信息
        
#         # ResNet18 特征
#         rgb_frame = frame[..., ::-1]  # BGR转RGB
#         resnet_input = preprocess(rgb_frame)
#         with torch.no_grad():
#             resnet_features = resnet(resnet_input.unsqueeze(0))
#             resnet_features = resnet_features.squeeze().cpu().numpy()
#         #print(f"ResNet features shape: {resnet_features.shape}")  # 调试信息
        
#         # 将所有特征拼接为一个向量
#         feature_vector = np.concatenate([
#             eye_state,
#             mouth_state,
#             head_pose,
#             emotion_state,
#             resnet_features
#         ])
#         #print(f"Final feature vector shape: {feature_vector.shape}")  # 调试信息
#         return feature_vector# 521
    
def extract_handcrafted_features(frame, face_detector, landmark_predictor):
    """
    提取手工特征（9维）：眼睛状态（2）、嘴巴状态（2）、头部姿态（3）、情绪（2）
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    if len(faces) == 0:
        return np.zeros(9)  # 如果没有检测到人脸，返回默认特征
    
    for face in faces:
        shape = landmark_predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # 眼睛状态
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        eye_ratio = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        eye_state = np.array([1, 0]) if eye_ratio < 0.2 else np.array([0, 1])
        
        # 嘴巴状态
        mouth = shape[48:60]
        mouth_ratio = mouth_aspect_ratio(mouth)
        mouth_state = np.array([1, 0]) if mouth_ratio > 0.5 else np.array([0, 1])
        
        # 头部姿态
        head_pose = estimate_head_pose(shape)
        
        # 表情识别
        analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        emotion = analysis[0]["dominant_emotion"]
        emotion_state = np.array([1, 0]) if emotion in ["happy", "neutral"] else np.array([0, 1])
        
        return np.concatenate([eye_state, mouth_state, head_pose, emotion_state])
    
    return np.zeros(9)  # 备用返回值

def extract_resnet_features(frame, model, device):
    """
    使用 ResNet 提取深度特征（512 维）。
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frame_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(frame_tensor)
    return features.cpu().numpy().flatten()

def extract_features(frame, face_detector, landmark_predictor, model, device):
    """
    结合手工特征（9维）和 ResNet 特征（512维）。
    """
    handcrafted_features = extract_handcrafted_features(frame, face_detector, landmark_predictor)
    resnet_features = extract_resnet_features(frame, model, device)
    return np.concatenate([handcrafted_features, resnet_features])


def create_dataset(video_path, output_csv_path, label):
    """
    创建数据集
    :param video_path: 视频文件路径
    :param output_csv_path: 输出 CSV 文件路径
    :param label: 标签（0 或 1）
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    feature_vectors = []
    labels = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = total_frames  # 处理整个视频

    print(f"Processing video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= end_frame:
            break

        # 提取特征
        feature_vector = extract_features(frame)

        # 检查特征向量的形状是否正确
        if feature_vector.shape[0] != 521:  # 2 (eye) + 2 (mouth) + 3 (head) + 2 (emotion) + 512 (resnet)
            print(f"Warning: Invalid feature vector shape at frame {frame_count}. Skipping this frame.")
            frame_count += 1
            continue

        # 保存特征和标签
        feature_vectors.append(feature_vector)
        labels.append(label)

        frame_count += 1
        if frame_count % fps == 0:
            print(f"Processed {frame_count // fps} seconds of video.")

    # 将特征和标签保存为 CSV 文件
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # 写入表头
        header = ["frame_number", "label"]
        header += [f"eye_state_{i}" for i in range(2)]  # 眼睛状态 (2维)
        header += [f"mouth_state_{i}" for i in range(2)]  # 嘴巴状态 (2维)
        header += [f"head_pose_{i}" for i in range(3)]  # 头部姿态 (3维)
        header += [f"emotion_state_{i}" for i in range(2)]  # 情绪状态 (2维)
        header += [f"resnet_feature_{i}" for i in range(512)]  # ResNet 特征 (512维)
        writer.writerow(header)
        
        # 写入数据
        for i, (feature_vector, label) in enumerate(zip(feature_vectors, labels)):
            row = [i, label]  # 帧编号和标签
            row += feature_vector.tolist()  # 特征向量
            writer.writerow(row)

    cap.release()
    print(f"Dataset creation completed. Data saved to {output_csv_path}.")

def process_videos(video_dir, output_dir):
    """
    批量处理视频
    :param video_dir: 视频文件目录
    :param output_dir: 输出 CSV 文件目录
    """
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    # 获取所有视频文件并按名称排序
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    num_videos = len(video_files)
    num_non_fatigue = int(num_videos * 2 / 3)  # 前 2/3 的视频为非疲劳

    for i, video_name in enumerate(video_files):
        video_path = os.path.join(video_dir, video_name)
        output_csv_path = os.path.join(output_dir, video_name.replace(".mp4", ".csv"))

        # 分配标签
        label = 0 if i < num_non_fatigue else 1  # 前 2/3 为非疲劳，后 1/3 为疲劳

        print(f"Processing video: {video_name}, Label: {label}")
        create_dataset(video_path, output_csv_path, label=label)

# 示例用法
if __name__ == "__main__":
    # video_dir = "E:/疲劳检测数据集/XQY/exp01/visible"  # 视频文件目录
    # output_dir = "data/csv/XQY"  # 输出 CSV 文件目录
    # process_videos(video_dir, output_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    frame = cv2.imread("example.png")
    feature_vector = extract_features(frame, face_detector, landmark_predictor, resnet, device)
    print(feature_vector.shape)  # 9+512=521 维特征