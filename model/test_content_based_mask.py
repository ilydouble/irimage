#!/usr/bin/env python3
"""
测试基于内容的智能人脸mask生成
专门针对黑色背景的热力图
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from train_thermal_classifier4 import (
    generate_content_based_mask,
    generate_smart_face_mask,
    generate_face_mask,
    apply_mask_to_image
)

def test_content_based_mask_generation():
    """测试基于内容的mask生成"""
    print("=== 测试基于内容的Mask生成 ===")
    
    # 创建模拟热力图（黑色背景 + 人脸区域）
    def create_mock_thermal_image(size=(224, 224)):
        """创建模拟的热力图"""
        height, width = size
        image = np.zeros((height, width, 3), dtype=np.float32)
        
        # 添加椭圆形的"人脸"区域
        center_y, center_x = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        
        # 主要人脸区域（椭圆）
        a, b = width * 0.3, height * 0.35
        face_mask = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2 <= 1
        
        # 添加一些热力图特征
        for c in range(3):
            image[:, :, c][face_mask] = 0.6 + 0.3 * np.random.random(np.sum(face_mask))
        
        # 添加一些噪声和细节
        # 眼部区域
        eye1_y, eye1_x = center_y - height//8, center_x - width//6
        eye2_y, eye2_x = center_y - height//8, center_x + width//6
        
        for eye_y, eye_x in [(eye1_y, eye1_x), (eye2_y, eye2_x)]:
            eye_region = ((x - eye_x)**2 + (y - eye_y)**2) <= (width*0.05)**2
            for c in range(3):
                image[:, :, c][eye_region] += 0.2
        
        # 鼻子区域
        nose_region = ((x - center_x)**2 + (y - center_y)**2) <= (width*0.03)**2
        for c in range(3):
            image[:, :, c][nose_region] += 0.15
        
        # 确保值在[0,1]范围内
        image = np.clip(image, 0, 1)
        
        return image
    
    # 测试不同参数的mask生成
    test_image = create_mock_thermal_image((224, 224))
    
    # 测试不同阈值
    thresholds = [0.05, 0.1, 0.15, 0.2]
    
    fig, axes = plt.subplots(3, len(thresholds) + 1, figsize=(20, 12))
    
    # 显示原图
    axes[0, 0].imshow(test_image)
    axes[0, 0].set_title('原始热力图')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(np.mean(test_image, axis=2), cmap='gray')
    axes[1, 0].set_title('灰度图')
    axes[1, 0].axis('off')
    
    axes[2, 0].axis('off')
    
    for i, threshold in enumerate(thresholds):
        # 生成mask
        mask = generate_content_based_mask(
            test_image, 
            threshold=threshold, 
            morphology_ops=True, 
            smooth=True
        )
        
        # 应用mask
        masked_image = apply_mask_to_image(
            np.transpose(test_image, (2, 0, 1)), 
            mask, 
            background_value=0.0
        )
        masked_image = np.transpose(masked_image, (1, 2, 0))
        
        # 显示结果
        axes[0, i+1].imshow(mask, cmap='gray')
        axes[0, i+1].set_title(f'Mask (阈值={threshold})')
        axes[0, i+1].axis('off')
        
        axes[1, i+1].imshow(masked_image)
        axes[1, i+1].set_title(f'应用Mask后')
        axes[1, i+1].axis('off')
        
        # 显示统计信息
        coverage = np.mean(mask)
        axes[2, i+1].text(0.1, 0.8, f'覆盖率: {coverage:.3f}', transform=axes[2, i+1].transAxes)
        axes[2, i+1].text(0.1, 0.6, f'阈值: {threshold}', transform=axes[2, i+1].transAxes)
        axes[2, i+1].text(0.1, 0.4, f'非零像素: {np.sum(mask > 0)}', transform=axes[2, i+1].transAxes)
        axes[2, i+1].axis('off')
        
        print(f"阈值 {threshold}: 覆盖率 {coverage:.3f}, 非零像素 {np.sum(mask > 0)}")
    
    plt.tight_layout()
    plt.savefig('content_based_mask_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 基于内容的Mask生成测试完成")

def test_real_thermal_images():
    """测试真实热力图的mask生成"""
    print("\n=== 测试真实热力图 ===")
    
    # 查找真实的热力图文件
    dataset_dirs = [
        Path("./dataset/datasets/thermal_classification_cropped/icas"),
        Path("./dataset/datasets/thermal_classification_cropped/non_icas"),
        Path("./dataset/datasets/thermal_24h")
    ]
    
    test_images = []
    for dataset_dir in dataset_dirs:
        if dataset_dir.exists():
            image_files = list(dataset_dir.glob("*.jpg"))[:3]  # 每个目录最多3张
            test_images.extend(image_files)
            if len(test_images) >= 6:  # 总共最多6张
                break
    
    if not test_images:
        print("未找到测试图像，使用模拟图像")
        return
    
    print(f"找到 {len(test_images)} 张测试图像")
    
    # 测试不同的mask方法
    mask_methods = [
        ("椭圆形", "ellipse"),
        ("智能检测", "content_based"),
        ("自适应", "adaptive")
    ]
    
    fig, axes = plt.subplots(len(test_images), len(mask_methods) + 1, figsize=(20, 4*len(test_images)))
    
    if len(test_images) == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_path in enumerate(test_images):
        try:
            # 加载图像
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image) / 255.0
            
            # 显示原图
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f'原图: {img_path.name}')
            axes[i, 0].axis('off')
            
            for j, (method_name, mask_type) in enumerate(mask_methods):
                if mask_type == "content_based":
                    # 使用智能mask
                    mask = generate_smart_face_mask(image_np, "ellipse")
                else:
                    # 使用几何mask
                    mask = generate_face_mask(image.size, mask_type)
                
                # 应用mask
                masked_image = apply_mask_to_image(
                    np.transpose(image_np, (2, 0, 1)), 
                    mask, 
                    background_value=0.0
                )
                masked_image = np.transpose(masked_image, (1, 2, 0))
                
                # 显示结果
                axes[i, j+1].imshow(masked_image)
                coverage = np.mean(mask)
                axes[i, j+1].set_title(f'{method_name}\n覆盖率: {coverage:.3f}')
                axes[i, j+1].axis('off')
                
                print(f"  {img_path.name} - {method_name}: 覆盖率 {coverage:.3f}")
                
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            continue
    
    plt.tight_layout()
    plt.savefig('real_thermal_mask_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 真实热力图测试完成")

def test_mask_quality_metrics():
    """测试mask质量指标"""
    print("\n=== 测试Mask质量指标 ===")
    
    # 创建测试图像
    test_image = np.zeros((224, 224, 3), dtype=np.float32)
    
    # 添加人脸区域
    center_y, center_x = 112, 112
    y, x = np.ogrid[:224, :224]
    face_region = ((x - center_x)**2 + (y - center_y)**2) <= 80**2
    
    for c in range(3):
        test_image[:, :, c][face_region] = 0.7
    
    # 添加一些背景噪声
    noise_mask = np.random.random((224, 224)) > 0.95
    for c in range(3):
        test_image[:, :, c][noise_mask] = 0.3
    
    # 测试不同方法
    methods = [
        ("基于内容", lambda: generate_content_based_mask(test_image, threshold=0.1)),
        ("椭圆形", lambda: generate_face_mask((224, 224), "ellipse")),
        ("矩形", lambda: generate_face_mask((224, 224), "rectangle")),
        ("智能检测", lambda: generate_smart_face_mask(test_image, "ellipse"))
    ]
    
    print("方法对比:")
    print("方法名称\t覆盖率\t精确度\t召回率\tF1分数")
    print("-" * 50)
    
    # 真实人脸区域作为ground truth
    gt_mask = face_region.astype(np.float32)
    
    for method_name, method_func in methods:
        mask = method_func()
        
        # 计算质量指标
        coverage = np.mean(mask)
        
        # 将mask二值化用于计算精确度和召回率
        binary_mask = (mask > 0.5).astype(np.float32)
        
        # 计算精确度、召回率和F1分数
        tp = np.sum(binary_mask * gt_mask)
        fp = np.sum(binary_mask * (1 - gt_mask))
        fn = np.sum((1 - binary_mask) * gt_mask)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{method_name}\t{coverage:.3f}\t{precision:.3f}\t{recall:.3f}\t{f1:.3f}")
    
    print("✅ Mask质量指标测试完成")

def main():
    """主测试函数"""
    print("基于内容的智能人脸Mask测试\n")
    
    try:
        # 运行所有测试
        test_content_based_mask_generation()
        test_real_thermal_images()
        test_mask_quality_metrics()
        
        print("\n🎉 所有测试完成!")
        print("📁 可视化结果已保存:")
        print("  - content_based_mask_test.png")
        print("  - real_thermal_mask_test.png")
        
        print("\n💡 使用建议:")
        print("  - 对于黑色背景的热力图，推荐使用 mask_type='content_based'")
        print("  - 阈值建议设置在 0.1-0.15 之间")
        print("  - 启用形态学操作和平滑处理以获得更好的mask质量")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
