import cv2
import mediapipe as mp
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def analyze_facial_features_from_image(image_path):
    """
    分析给定图像路径中的人脸特征并绘制检测框。
    """
    # 初始化 MediaPipe 人脸网格
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # 读取并处理图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法加载图像，请检查路径： {image_path}")
        return

    # 创建一个副本用于绘制
    result_image = image.copy()
    
    # 将 BGR 图像转换为 RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 运行人脸网格检测
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print("未检测到人脸。")
        return

    # 假定图像中只有一张脸
    face_landmarks = results.multi_face_landmarks[0]
    ih, iw, _ = image.shape

    # 将关键点坐标从归一化值转换为像素坐标
    landmarks_pixels = np.array([(int(pt.x * iw), int(pt.y * ih)) for pt in face_landmarks.landmark])

    # 定义各个区域的关键点索引和颜色
    regions = {
        'left_eye': ([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], (0, 255, 0)),
        'right_eye': ([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398], (0, 255, 0)),
        'nose': ([1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 360, 279], (255, 0, 0)),
        'left_cheek': ([116, 117, 118, 119, 120, 121, 128, 126, 142, 36, 205, 206, 207, 213, 192, 147], (0, 0, 255)),
        'right_cheek': ([345, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410, 454, 323, 361, 340], (0, 0, 255)),
        'mouth': ([61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 269, 270, 267, 271, 272], (255, 0, 255))
    }
    
    # 将图像转换为灰度图以进行 GLCM 计算
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 为每个区域提取特征并绘制检测框
    for region_name, (indices, color) in regions.items():
        print(f"\n--- 正在分析区域: {region_name} ---")

        # 获取区域的边界框
        points = landmarks_pixels[indices]
        x, y, w, h = cv2.boundingRect(points)
        
        # 确保边界框不为空
        if w == 0 or h == 0:
            print(f"区域 '{region_name}' 的边界框为空，跳过此区域。")
            continue

        # 绘制检测框
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        
        # 添加标签
        cv2.putText(result_image, region_name, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 提取区域进行特征分析
        region_of_interest = gray_image[y:y+h, x:x+w]
        
        if region_of_interest.size == 0:
            print(f"提取的区域 '{region_name}' 为空，跳过此区域。")
            continue
            
        # 提取温度特征
        min_temp = np.min(region_of_interest)
        max_temp = np.max(region_of_interest)
        mean_temp = np.mean(region_of_interest)

        print(f"温度特征:")
        print(f"  均值: {mean_temp:.2f}")
        print(f"  最大值: {max_temp}")
        print(f"  最小值: {min_temp}")

        # 提取纹理特征
        if region_of_interest.shape[0] > 1 and region_of_interest.shape[1] > 1:
            glcm = graycomatrix(region_of_interest, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]

            print("纹理特征 (GLCM):")
            print(f"  对比度: {contrast:.4f}")
            print(f"  相异性: {dissimilarity:.4f}")
            print(f"  同质性: {homogeneity:.4f}")
            print(f"  能量: {energy:.4f}")
            print(f"  相关性: {correlation:.4f}")
        else:
            print("纹理特征 (GLCM): 区域太小，无法计算。")

    # 保存和显示结果
    output_path = image_path.replace('.jpg', '_detected.jpg')
    cv2.imwrite(output_path, result_image)
    print(f"\n检测结果已保存到: {output_path}")
    
    # 显示图像
    cv2.imshow('Face Detection Results', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- 主程序入口 ---
if __name__ == '__main__':
    image_file = 'data/CK061-1.jpg' 
    analyze_facial_features_from_image(image_file)
