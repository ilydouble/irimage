import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from typing import Tuple, Optional

# ======================================================================================
# 模块一：人脸检测与分割 (Facial Detection and Splitting)
# ======================================================================================

def preprocess_and_split_face(
    image_path: str, 
    output_size: Tuple[int, int] = (224, 112)
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    加载图像，检测人脸，沿中心线分割，并为神经网络准备左右半脸张量。

    Args:
        image_path (str): 图像文件路径。
        output_size (Tuple[int, int]): 每个半脸调整后的尺寸 (高度, 宽度)。

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor]]: 一个包含左脸和右脸张量的元组，
                                                     如果未检测到人脸则返回 None。
    """
    # 初始化 MediaPipe 人脸网格
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法加载图像: {image_path}")
        return None

    # 运行人脸检测
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        print("未在此图像中检测到人脸。")
        return None

    # 获取关键点和图像尺寸
    face_landmarks = results.multi_face_landmarks[0]
    ih, iw, _ = image.shape
    landmarks_pixels = np.array([(pt.x * iw, pt.y * ih) for pt in face_landmarks.landmark])

    # 1. 计算人脸包围框
    x_min = int(np.min(landmarks_pixels[:, 0]))
    x_max = int(np.max(landmarks_pixels[:, 0]))
    y_min = int(np.min(landmarks_pixels[:, 1]))
    y_max = int(np.max(landmarks_pixels[:, 1]))
    
    # 2. 沿垂直中轴线分割人脸
    # 我们使用面部中线上的关键点来确定精确的中心 (例如，鼻子和下巴上的点)
    # MediaPipe 关键点索引: 鼻梁[168, 10], 鼻尖[1], 下巴[152]
    centerline_x = int(np.mean([landmarks_pixels[i, 0] for i in [168, 10, 1, 152]]))

    # 裁剪左右脸
    # 注意：在图像中，左脸在右侧，右脸在左侧
    right_face_img = image[y_min:y_max, x_min:centerline_x]
    left_face_img = image[y_min:y_max, centerline_x:x_max]

    if left_face_img.size == 0 or right_face_img.size == 0:
        print("错误: 分割后的一半或多半脸为空。")
        return None

    # 3. 预处理：将图像转换为PyTorch张量
    # 定义一个转换流程：转为PIL图像 -> 调整大小 -> 转为张量 -> 归一化
    # 热图是单通道的，但像ResNet这样的预训练模型期望3通道输入。
    # 我们将灰度图复制到3个通道中。
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(output_size),
        T.Grayscale(num_output_channels=3), # 适配ResNet等模型的输入
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    left_tensor = transform(left_face_img)
    right_tensor = transform(right_face_img)
    
    return left_tensor, right_tensor

# ======================================================================================
# 模块二：孪生卷积神经网络 (Siamese CNN)
# ======================================================================================

class SiameseNetwork(nn.Module):
    """
    一个孪生网络，使用共享权重的CNN从输入对中提取特征向量。
    """
    def __init__(self, backbone: str = 'resnet18'):
        super(SiameseNetwork, self).__init__()
        
        # 1. 加载一个预训练的CNN模型作为主干
        if backbone == 'resnet18':
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_features = self.cnn.fc.in_features
            # 移除原始的分类层，我们想要它之前的特征
            self.cnn.fc = nn.Identity()
        elif backbone == 'mobilenet_v2':
            self.cnn = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            num_features = self.cnn.classifier[1].in_features
            # 移除分类器
            self.cnn.classifier = nn.Identity()
        else:
            raise ValueError("不支持的主干网络。请选择 'resnet18' 或 'mobilenet_v2'。")

        print(f"已加载 {backbone} 作为主干网络。特征向量维度: {num_features}")

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """通过一个分支处理单个输入。"""
        return self.cnn(x)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理两个输入，并返回它们各自的特征向量。
        在推理（非训练）模式下，我们通常一次只处理一对。
        """
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# ======================================================================================
# 模块三：主执行流程 (Main Execution Flow)
# ======================================================================================

def get_asymmetry_vector(model: SiameseNetwork, image_path: str) -> Optional[np.ndarray]:
    """
    处理单张图像，提取左右脸，并通过孪生网络计算不对称性向量。

    Args:
        model (SiameseNetwork): 已初始化的孪生网络模型。
        image_path (str): 图像文件路径。

    Returns:
        Optional[np.ndarray]: 代表面部不对称性的特征向量，如果出错则返回 None。
    """
    # 1. 预处理图像以获取左右脸张量
    face_halves = preprocess_and_split_face(image_path)
    if face_halves is None:
        return None
    
    left_face_tensor, right_face_tensor = face_halves

    # 2. 准备输入模型的批次
    # 模型期望一个批次作为输入，所以我们添加一个批次维度 (B, C, H, W)
    left_batch = left_face_tensor.unsqueeze(0)
    right_batch = right_face_tensor.unsqueeze(0)

    # 3. 通过模型进行前向传播以获取特征向量
    model.eval()  # 设置为评估模式
    with torch.no_grad(): # 在推理时不需要计算梯度
        feature_vec_left, feature_vec_right = model(left_batch, right_batch)

    # 4. 计算不对称性向量 (向量差)
    asymmetry_vector = feature_vec_left - feature_vec_right
    
    # 5. 返回NumPy数组格式的结果
    return asymmetry_vector.squeeze().cpu().numpy()


# --- 主程序入口 ---
if __name__ == '__main__':
    # 请将 'path/to/your/image.jpg' 替换为您的红外图像文件的实际路径
    # 这里我们使用您提供的示例图片名称
    image_file = 'thermal_face_image.jpg'
    
    # 初始化孪生网络模型
    # 您可以选择 'resnet18' 或 'mobilenet_v2'
    siamese_model = SiameseNetwork(backbone='resnet18')
    
    print("\n--- 正在计算面部不对称性特征向量 ---")
    # 获取不对称性向量
    asymmetry_features = get_asymmetry_vector(siamese_model, image_file)
    
    if asymmetry_features is not None:
        print("\n--- 分析完成 ---")
        print(f"成功提取不对称性特征向量！")
        print(f"向量维度: {asymmetry_features.shape}")
        # 打印向量的前10个元素作为示例
        print(f"特征向量 (前10个元素): \n{asymmetry_features[:10]}")
        print("\n注意：这是一个未经训练的模型的输出。该向量需要通过在带标签数据集上进行训练才能获得实际意义。")
    else:
        print("\n--- 分析失败 ---")
        print("无法为该图像生成不对称性向量。请检查上面的错误信息。")