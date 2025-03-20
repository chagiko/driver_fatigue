import cv2
import dlib
import numpy as np
from deepface import DeepFace
from imutils import face_utils
import torch
from torchvision import models, transforms
import random
import csv
import random
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
def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    # 初始化默认特征向量
    default_features = {
        "eye_state": np.zeros(2),  # 眼睛状态 (睁眼/闭眼)
        "mouth_state": np.zeros(2),  # 嘴巴状态 (张开/闭合)
        "head_pose": np.zeros(3),  # 头部姿态 (pitch, yaw, roll)
        "emotion": np.zeros(2),  # 情绪 (积极/消极)
        "resnet_features": np.zeros(512)  # ResNet18 特征 (512维)
    }
    
    if len(faces) == 0:
        # 如果没有检测到人脸，返回默认特征
        print("No face detected. Returning default features.")
        feature_vector = np.concatenate([
            default_features["eye_state"],
            default_features["mouth_state"],
            default_features["head_pose"],
            default_features["emotion"],
            default_features["resnet_features"]
        ])
        print(f"Default feature vector shape: {feature_vector.shape}")
        return feature_vector
    
    for face in faces:
        shape = landmark_predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # 眼睛状态
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        eye_ratio = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        eye_state = np.array([1, 0]) if eye_ratio < 0.2 else np.array([0, 1])
        #print(f"Eye state shape: {eye_state.shape}")  # 调试信息
        
        # 嘴巴状态
        mouth = shape[48:60]
        mouth_ratio = mouth_aspect_ratio(mouth)
        mouth_state = np.array([1, 0]) if mouth_ratio > 0.5 else np.array([0, 1])
        #print(f"Mouth state shape: {mouth_state.shape}")  # 调试信息
        
        # 头部姿态
        head_pose = estimate_head_pose(shape)
        #print(f"Head pose shape: {head_pose.shape}")  # 调试信息
        
        # 表情识别
        analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        emotion = analysis[0]["dominant_emotion"]
        emotion_state = np.array([1, 0]) if emotion in ["happy", "neutral"] else np.array([0, 1])
        #print(f"Emotion state shape: {emotion_state.shape}")  # 调试信息
        
        # ResNet18 特征
        rgb_frame = frame[..., ::-1]  # BGR转RGB
        resnet_input = preprocess(rgb_frame)
        with torch.no_grad():
            resnet_features = resnet(resnet_input.unsqueeze(0))
            resnet_features = resnet_features.squeeze().cpu().numpy()
        #print(f"ResNet features shape: {resnet_features.shape}")  # 调试信息
        
        # 将所有特征拼接为一个向量
        feature_vector = np.concatenate([
            eye_state,
            mouth_state,
            head_pose,
            emotion_state,
            resnet_features
        ])
        #print(f"Final feature vector shape: {feature_vector.shape}")  # 调试信息
        return feature_vector# 521

def create_dataset(video_path, output_csv_path, duration=100):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    feature_vectors = []
    labels = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(fps * duration, total_frames)

    print(f"Video duration: {total_frames // fps} seconds")

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

        # 随机生成标签（0 或 1）
        label = random.choice([0, 1])  # 0: 非疲劳, 1: 疲劳

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
        header = ["frame_number"]
        header += [f"eye_state_{i}" for i in range(2)]  # 眼睛状态 (2维)
        header += [f"mouth_state_{i}" for i in range(2)]  # 嘴巴状态 (2维)
        header += [f"head_pose_{i}" for i in range(3)]  # 头部姿态 (3维)
        header += [f"emotion_state_{i}" for i in range(2)]  # 情绪状态 (2维)
        header += [f"resnet_feature_{i}" for i in range(512)]  # ResNet 特征 (512维)
        header += ["label"]  # 标签放在最后一列
        writer.writerow(header)
        
        # 写入数据
        for i, (feature_vector, label) in enumerate(zip(feature_vectors, labels)):
            row = [i]  # 帧编号
            row += feature_vector.tolist()  # 特征向量
            row.append(label)  # 标签放在最后一列
            writer.writerow(row)

    cap.release()
    print(f"Dataset creation completed. Data saved to {output_csv_path}.")


if __name__ == "__main__":
    # 视频路径和输出的NPY文件路径
    video_path = "visible.mp4"
    output_npy_path = "output_features2.csv"
    
    # 创建数据集
    create_dataset(video_path, output_npy_path, duration=200)